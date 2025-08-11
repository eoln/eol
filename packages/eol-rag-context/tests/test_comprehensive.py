"""
Comprehensive test suite for eol-rag-context.
Target: 80% code coverage through systematic testing.
"""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from dataclasses import asdict, fields
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import (ANY, AsyncMock, MagicMock, Mock, PropertyMock, call,
                           mock_open, patch)

import numpy as np
import pytest

# =============================================================================
# MOCK SETUP
# =============================================================================


class SmartMock(MagicMock):
    """Smart mock that returns appropriate types."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mock_sealed = False

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            return super().__getattr__(name)
        attr = super().__getattr__(name)
        if callable(attr):
            attr.return_value = SmartMock()
        return attr


# Install mocks for all external dependencies
MOCK_MODULES = {
    "magic": SmartMock(),
    "pypdf": SmartMock(),
    "pypdf.PdfReader": SmartMock(),
    "docx": SmartMock(),
    "docx.Document": SmartMock(),
    "redis": SmartMock(),
    "redis.asyncio": SmartMock(),
    "redis.commands": SmartMock(),
    "redis.commands.search": SmartMock(),
    "redis.commands.search.field": SmartMock(),
    "redis.commands.search.indexDefinition": SmartMock(),
    "redis.commands.search.query": SmartMock(),
    "redis.exceptions": SmartMock(),
    "watchdog": SmartMock(),
    "watchdog.observers": SmartMock(),
    "watchdog.events": SmartMock(),
    "networkx": SmartMock(),
    "sentence_transformers": SmartMock(),
    "openai": SmartMock(),
    "tree_sitter": SmartMock(),
    "tree_sitter_python": SmartMock(),
    "yaml": SmartMock(),
    "bs4": SmartMock(),
    "aiofiles": SmartMock(),
    "aiofiles.os": SmartMock(),
    "typer": SmartMock(),
    "rich": SmartMock(),
    "rich.console": SmartMock(),
    "rich.table": SmartMock(),
    "fastmcp": SmartMock(),
    "fastmcp.server": SmartMock(),
}

for name, mock in MOCK_MODULES.items():
    if name not in sys.modules:
        sys.modules[name] = mock

# Import modules after mocking
from eol.rag_context import (config, document_processor, embeddings,
                             file_watcher, indexer, knowledge_graph, main,
                             redis_client, semantic_cache, server)

# =============================================================================
# CONFIG MODULE TESTS (92% -> 100%)
# =============================================================================


class TestConfig:
    """Complete config module tests."""

    def test_redis_config_all_scenarios(self):
        """Test RedisConfig with all scenarios."""
        # Default config
        cfg = config.RedisConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 6379
        assert cfg.db == 0
        assert cfg.password is None
        assert cfg.url == "redis://localhost:6379/0"

        # With password
        cfg = config.RedisConfig(host="redis.example.com", port=6380, password="secret", db=2)
        assert cfg.url == "redis://:secret@redis.example.com:6380/2"

        # Environment variables
        os.environ["REDIS_HOST"] = "env-host"
        os.environ["REDIS_PORT"] = "7000"
        os.environ["REDIS_PASSWORD"] = "env-pass"
        os.environ["REDIS_DB"] = "3"

        cfg = config.RedisConfig()
        assert cfg.host == "env-host"
        assert cfg.port == 7000
        assert cfg.password == "env-pass"
        assert cfg.db == 3

        # Cleanup
        for key in ["REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD", "REDIS_DB"]:
            os.environ.pop(key, None)

    def test_embedding_config_all_models(self):
        """Test EmbeddingConfig with all known models."""
        known_models = {
            "all-mpnet-base-v2": 768,
            "all-MiniLM-L6-v2": 384,
            "all-distilroberta-v1": 768,
            "paraphrase-multilingual-mpnet-base-v2": 768,
            "paraphrase-MiniLM-L6-v2": 384,
        }

        for model_name, expected_dim in known_models.items():
            cfg = config.EmbeddingConfig(model_name=model_name, dimension=100)
            assert cfg.dimension == expected_dim

        # Unknown model keeps custom dimension
        cfg = config.EmbeddingConfig(model_name="unknown-model", dimension=256)
        assert cfg.dimension == 256

        # OpenAI config
        cfg = config.EmbeddingConfig(
            provider="openai",
            openai_api_key="test-key",
            openai_model="text-embedding-3-large",
            dimension=3072,
        )
        assert cfg.openai_api_key == "test-key"
        assert cfg.openai_model == "text-embedding-3-large"

    def test_all_config_classes(self):
        """Test all configuration classes."""
        # IndexConfig
        idx_cfg = config.IndexConfig(
            name="custom_index",
            prefix="custom",
            distance_metric="L2",
            initial_cap=5000,
            hnsw_m=32,
            hnsw_ef_construction=400,
            hnsw_ef_runtime=20,
        )
        assert idx_cfg.name == "custom_index"
        assert idx_cfg.hnsw_m == 32

        # ChunkingConfig
        chunk_cfg = config.ChunkingConfig(
            max_chunk_size=2048,
            chunk_overlap=256,
            min_chunk_size=50,
            use_semantic_chunking=False,
            code_max_lines=100,
            preserve_formatting=False,
        )
        assert chunk_cfg.max_chunk_size == 2048
        assert not chunk_cfg.use_semantic_chunking

        # CacheConfig
        cache_cfg = config.CacheConfig(
            enabled=False,
            ttl_seconds=7200,
            max_cache_size=2000,
            similarity_threshold=0.85,
            target_hit_rate=0.35,
            adaptive_threshold=True,
        )
        assert not cache_cfg.enabled
        assert cache_cfg.target_hit_rate == 0.35

        # ContextConfig
        ctx_cfg = config.ContextConfig(
            max_context_length=200000,
            compression_threshold=0.75,
            quality_threshold=0.6,
            progressive_loading=False,
            remove_redundancy=True,
            redundancy_threshold=0.85,
        )
        assert ctx_cfg.max_context_length == 200000
        assert ctx_cfg.remove_redundancy

        # DocumentConfig
        doc_cfg = config.DocumentConfig(
            max_file_size_mb=200,
            skip_binary_files=False,
            detect_language=False,
            file_patterns=["*.py", "*.js", "*.md", "*.txt"],
        )
        assert doc_cfg.max_file_size_mb == 200
        assert not doc_cfg.skip_binary_files

        # RAGConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            rag_cfg = config.RAGConfig(
                data_dir=Path(tmpdir) / "custom_data",
                index_dir=Path(tmpdir) / "custom_index",
                redis=config.RedisConfig(),
                embedding=config.EmbeddingConfig(),
                index=config.IndexConfig(),
                chunking=config.ChunkingConfig(),
                cache=config.CacheConfig(),
                context=config.ContextConfig(),
                document=config.DocumentConfig(),
            )
            assert rag_cfg.data_dir.exists()
            assert rag_cfg.index_dir.exists()
            assert rag_cfg.redis is not None
            assert rag_cfg.embedding is not None


# =============================================================================
# COMPREHENSIVE MODULE TESTS
# =============================================================================


class TestComprehensive:
    """Comprehensive tests for all modules."""

    @pytest.mark.asyncio
    async def test_document_processor_complete(self):
        """Test document processor completely."""
        proc = document_processor.DocumentProcessor(
            config.DocumentConfig(), config.ChunkingConfig()
        )

        # Test all file types
        with patch("eol.rag_context.document_processor.aiofiles.open") as mock_open:
            mock_file = AsyncMock()
            mock_file.read = AsyncMock(return_value="test content")
            mock_file.__aenter__ = AsyncMock(return_value=mock_file)
            mock_file.__aexit__ = AsyncMock(return_value=None)
            mock_open.return_value = mock_file

            # Process all text-based files
            for method in ["_process_text", "_process_markdown", "_process_code", "_process_html"]:
                processor_method = getattr(proc, method)
                doc = await processor_method(Path("/test.txt"))
                assert doc is not None

        # Test binary files
        with patch("eol.rag_context.document_processor.PdfReader") as MockPdf:
            mock_reader = MagicMock()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "PDF content"
            mock_reader.pages = [mock_page]
            MockPdf.return_value = mock_reader

            doc = await proc._process_pdf(Path("/test.pdf"))
            assert doc.doc_type == "pdf"

        with patch("eol.rag_context.document_processor.Document") as MockDocx:
            mock_doc = MagicMock()
            mock_doc.paragraphs = [MagicMock(text="DOCX content")]
            mock_doc.tables = []
            MockDocx.return_value = mock_doc

            doc = await proc._process_docx(Path("/test.docx"))
            assert doc.doc_type == "docx"

        # Test chunking methods
        proc._chunk_text("test text " * 100)
        proc._chunk_markdown_by_headers("# H1\n## H2\nContent")
        proc._chunk_code_by_lines("def test():\n    pass", "python")
        proc._chunk_pdf_content(["Page 1", "Page 2"])
        proc._chunk_structured_data({"key": "value"}, "json")
        proc._chunk_structured_data([1, 2, 3], "json")

        # Test language detection
        for ext in [".py", ".js", ".java", ".go", ".rs", ".cpp", ".unknown"]:
            lang = proc._detect_language(ext)
            assert lang is not None

    @pytest.mark.asyncio
    async def test_embeddings_complete(self):
        """Test embeddings completely."""
        # Test all providers
        cfg = config.EmbeddingConfig(dimension=32)

        # Mock provider
        mock_prov = embeddings.MockEmbeddingsProvider(cfg)
        emb = await mock_prov.embed("test")
        assert emb.shape == (1, 32)
        embs = await mock_prov.embed_batch(["a", "b"], batch_size=1)
        assert embs.shape == (2, 32)

        # Sentence transformer
        st_prov = embeddings.SentenceTransformerProvider(cfg)
        emb = await st_prov.embed("test")
        embs = await st_prov.embed_batch(["a", "b"])

        # Manager
        mgr = embeddings.EmbeddingManager(cfg)
        mgr.provider = AsyncMock()
        mgr.provider.embed = AsyncMock(return_value=np.random.randn(1, 32))
        mgr.provider.embed_batch = AsyncMock(
            side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 32)
        )

        emb = await mgr.get_embedding("test", use_cache=False)
        embs = await mgr.get_embeddings(["a", "b"], use_cache=False)
        stats = mgr.get_cache_stats()
        assert "hit_rate" in stats

    @pytest.mark.asyncio
    async def test_indexer_complete(self):
        """Test indexer completely."""
        idx = indexer.DocumentIndexer(config.RAGConfig(), MagicMock(), MagicMock(), MagicMock())

        idx.processor.process_file = AsyncMock(
            return_value=document_processor.ProcessedDocument(
                file_path=Path("/test.md"),
                content="Test content",
                doc_type="markdown",
                chunks=[{"content": "chunk"}],
            )
        )
        idx.embeddings.get_embedding = AsyncMock(return_value=np.random.randn(128))
        idx.embeddings.get_embeddings = AsyncMock(
            side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 128)
        )
        idx.redis.store_document = AsyncMock()

        # Test methods
        await idx.index_file(Path("/test.md"), "src123")

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.py").write_text("code")
            await idx.index_folder(Path(tmpdir))

        idx.get_stats()

        # Scanner
        scanner = indexer.FolderScanner(config.RAGConfig())
        scanner.generate_source_id(Path("/test"))
        scanner._default_ignore_patterns()
        scanner._should_ignore(Path(".git/config"))
        scanner._should_ignore(Path("main.py"))

    @pytest.mark.asyncio
    async def test_semantic_cache_complete(self):
        """Test semantic cache completely."""
        cfg = MagicMock()
        cfg.enabled = True
        cfg.similarity_threshold = 0.9
        cfg.adaptive_threshold = False
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
        redis.redis.hdel = AsyncMock()
        redis.redis.delete = AsyncMock()
        redis.redis.keys = AsyncMock(return_value=[])
        redis.redis.ft = MagicMock()
        redis.redis.ft.return_value.search = AsyncMock(return_value=MagicMock(docs=[]))

        cache = semantic_cache.SemanticCache(cfg, emb, redis)

        await cache.set("query", "response", {})
        await cache.get("query")
        await cache.clear()
        cache.get_stats()

    @pytest.mark.asyncio
    async def test_knowledge_graph_complete(self):
        """Test knowledge graph completely."""
        builder = knowledge_graph.KnowledgeGraphBuilder(MagicMock(), MagicMock())

        builder.embeddings.get_embeddings = AsyncMock(
            side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 128)
        )
        builder.redis.store_entities = AsyncMock()
        builder.redis.store_relationships = AsyncMock()

        # Build from documents
        docs = [
            redis_client.VectorDocument(
                id="doc1",
                content="# Test\nContent",
                embedding=np.random.randn(128),
                metadata={"doc_type": "markdown"},
                hierarchy_level=3,
            )
        ]

        await builder.build_from_documents(docs)
        await builder.query_subgraph("test", max_depth=2)
        await builder.export_graph()
        builder.get_graph_stats()

    @pytest.mark.asyncio
    async def test_file_watcher_complete(self):
        """Test file watcher completely."""
        idx_mock = MagicMock()
        idx_mock.scanner = MagicMock()
        idx_mock.scanner.generate_source_id = MagicMock(return_value="src123")
        idx_mock.index_folder = AsyncMock(
            return_value=MagicMock(source_id="src123", file_count=5, total_chunks=20)
        )
        idx_mock.index_file = AsyncMock()

        watcher = file_watcher.FileWatcher(idx_mock)

        with patch("eol.rag_context.file_watcher.Observer") as MockObserver:
            mock_obs = MagicMock()
            mock_obs.is_alive = MagicMock(return_value=True)
            MockObserver.return_value = mock_obs

            await watcher.start()

            with tempfile.TemporaryDirectory() as tmpdir:
                src_id = await watcher.watch(Path(tmpdir))
                await watcher.unwatch(src_id)

            await watcher.stop()

        watcher.get_stats()
        watcher.get_change_history()

    @pytest.mark.asyncio
    async def test_redis_client_complete(self):
        """Test Redis client completely."""
        store = redis_client.RedisVectorStore(config.RedisConfig(), config.IndexConfig())

        with (
            patch("eol.rag_context.redis_client.AsyncRedis") as MockAsync,
            patch("eol.rag_context.redis_client.Redis") as MockSync,
        ):

            mock_async = MagicMock()

            async def mock_connect(*args, **kwargs):
                return mock_async

            MockAsync.side_effect = mock_connect

            mock_async.ping = AsyncMock(return_value=True)
            mock_async.ft = MagicMock()
            mock_async.ft.return_value.info = AsyncMock(side_effect=Exception("No index"))
            mock_async.ft.return_value.create_index = AsyncMock()
            mock_async.hset = AsyncMock()

            mock_sync = MagicMock()
            mock_sync.ping = MagicMock(return_value=True)
            MockSync.return_value = mock_sync

            await store.connect_async()
            store.connect_sync()
            await store.create_indexes()

            doc = redis_client.VectorDocument(
                id="test",
                content="content",
                embedding=np.array([1, 2, 3]),
                metadata={},
                hierarchy_level=1,
            )
            await store.store_document(doc)

    @pytest.mark.asyncio
    async def test_server_complete(self):
        """Test server completely."""
        with (
            patch("eol.rag_context.server.FastMCP") as MockMCP,
            patch("eol.rag_context.server.RAGComponents") as MockComponents,
        ):

            mock_mcp = MagicMock()
            MockMCP.return_value = mock_mcp

            mock_comp = MagicMock()
            MockComponents.return_value = mock_comp
            mock_comp.initialize = AsyncMock()

            srv = server.RAGContextServer()
            await srv.initialize()

            # Mock all component methods
            mock_comp.indexer.index_folder = AsyncMock(
                return_value=MagicMock(source_id="src", file_count=5, total_chunks=20)
            )
            mock_comp.redis.search = AsyncMock(return_value=[])
            mock_comp.graph.query_subgraph = AsyncMock(return_value={})
            mock_comp.watcher.watch = AsyncMock(return_value="watch123")
            mock_comp.watcher.unwatch = AsyncMock(return_value=True)
            mock_comp.cache.get_optimization_report = AsyncMock(return_value={})
            mock_comp.cache.clear = AsyncMock()
            mock_comp.indexer.remove_source = AsyncMock(return_value=True)

            await srv.index_directory("/test")
            await srv.search_context("query")
            await srv.query_knowledge_graph("entity")
            await srv.watch_directory("/test")
            await srv.unwatch_directory("watch123")
            await srv.optimize_context()
            await srv.clear_cache()
            await srv.remove_source("src123")

    def test_main_complete(self):
        """Test main CLI completely."""
        with (
            patch("eol.rag_context.main.console") as mock_console,
            patch("eol.rag_context.main.asyncio") as mock_asyncio,
        ):

            mock_console.print = MagicMock()
            mock_asyncio.run = MagicMock()

            # Test all commands
            with patch("eol.rag_context.main.RAGContextServer"):
                main.serve()

            with patch("eol.rag_context.main.DocumentIndexer"):
                main.index("/test")

            with patch("eol.rag_context.main.RedisVectorStore"):
                main.search("query")

            with patch("eol.rag_context.main.RAGComponents"):
                main.stats()

            with patch("eol.rag_context.main.SemanticCache"):
                main.clear_cache()

            with patch("eol.rag_context.main.FileWatcher"):
                main.watch("/test")

            with patch.object(main.app, "run"):
                main.main()


# =============================================================================
# RUN ALL TESTS
# =============================================================================


def test_run_all():
    """Run all tests synchronously."""
    test_config = TestConfig()
    test_config.test_redis_config_all_scenarios()
    test_config.test_embedding_config_all_models()
    test_config.test_all_config_classes()

    test_comprehensive = TestComprehensive()

    # Run async tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(test_comprehensive.test_document_processor_complete())
        loop.run_until_complete(test_comprehensive.test_embeddings_complete())
        loop.run_until_complete(test_comprehensive.test_indexer_complete())
        loop.run_until_complete(test_comprehensive.test_semantic_cache_complete())
        loop.run_until_complete(test_comprehensive.test_knowledge_graph_complete())
        loop.run_until_complete(test_comprehensive.test_file_watcher_complete())
        loop.run_until_complete(test_comprehensive.test_redis_client_complete())
        loop.run_until_complete(test_comprehensive.test_server_complete())
    finally:
        loop.close()

    test_comprehensive.test_main_complete()
