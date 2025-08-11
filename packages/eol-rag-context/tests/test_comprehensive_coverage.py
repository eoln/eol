"""
Comprehensive test suite to achieve 80% coverage for EOL RAG Context.
This file tests all untested code paths to reach the 80% coverage requirement.
"""

import asyncio
import hashlib
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest

# Mock external dependencies before importing our modules
sys.modules["magic"] = MagicMock()
sys.modules["pypdf"] = MagicMock()
sys.modules["docx"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["openai"] = MagicMock()
sys.modules["fastmcp"] = MagicMock()
sys.modules["fastmcp.server"] = MagicMock()
sys.modules["tree_sitter"] = MagicMock()
sys.modules["watchdog"] = MagicMock()
sys.modules["watchdog.observers"] = MagicMock()
sys.modules["watchdog.events"] = MagicMock()
sys.modules["networkx"] = MagicMock()
sys.modules["yaml"] = MagicMock()
sys.modules["bs4"] = MagicMock()
sys.modules["markdown"] = MagicMock()
sys.modules["typer"] = MagicMock()
sys.modules["rich"] = MagicMock()
sys.modules["rich.console"] = MagicMock()
sys.modules["rich.table"] = MagicMock()
sys.modules["gitignore_parser"] = MagicMock()

# Now import our modules
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


class TestRedisClientCoverage:
    """Achieve 80% coverage for redis_client.py"""

    def test_redis_connection_and_operations(self):
        """Test Redis connection and basic operations"""
        cfg = config.RedisConfig()
        idx_cfg = config.IndexConfig()

        with patch("eol.rag_context.redis_client.Redis") as MockRedis:
            mock_redis = MagicMock()
            MockRedis.return_value = mock_redis

            # Test RedisVectorStore initialization
            store = redis_client.RedisVectorStore(cfg, idx_cfg)
            assert store.redis_config == cfg
            assert store.index_config == idx_cfg

            # Test connect
            store.connect()
            MockRedis.assert_called_once()

            # Test create_hierarchical_indexes
            mock_redis.ft = MagicMock()
            mock_redis.ft.create_index = MagicMock()
            store.create_hierarchical_indexes(embedding_dim=768)

            # Test store_document
            doc_id = store.store_document(
                content="test content",
                embedding=[0.1] * 768,
                metadata={"source": "test.py"},
                hierarchy_level=3,
            )
            assert doc_id is not None

            # Test search_similar
            mock_redis.ft().search = MagicMock(return_value=MagicMock(docs=[]))
            results = store.search_similar([0.1] * 768, k=5, hierarchy_level=3)
            assert isinstance(results, list)

            # Test get_document
            mock_redis.hgetall = MagicMock(return_value={"content": "test"})
            doc = store.get_document("doc1")
            assert doc is not None

            # Test delete_document
            mock_redis.delete = MagicMock(return_value=1)
            result = store.delete_document("doc1")
            assert result is True

    @pytest.mark.asyncio
    async def test_async_operations(self):
        """Test async Redis operations"""
        cfg = config.RedisConfig()
        idx_cfg = config.IndexConfig()

        with patch("eol.rag_context.redis_client.AsyncRedis") as MockAsyncRedis:
            mock_async_redis = AsyncMock()
            MockAsyncRedis.return_value = mock_async_redis

            store = redis_client.RedisVectorStore(cfg, idx_cfg)

            # Test async connect
            await store.connect_async()
            MockAsyncRedis.assert_called_once()

            # Test async search
            mock_async_redis.ft = MagicMock()
            mock_async_redis.ft().search = AsyncMock(return_value=MagicMock(docs=[]))
            results = await store.search_similar_async([0.1] * 768, k=5)
            assert isinstance(results, list)

            # Test close
            await store.close()
            mock_async_redis.close.assert_called_once()


class TestDocumentProcessorCoverage:
    """Achieve 80% coverage for document_processor.py"""

    @pytest.mark.asyncio
    async def test_process_documents(self):
        """Test document processing for various formats"""
        doc_cfg = config.DocumentConfig()
        chunk_cfg = config.ChunkingConfig()

        processor = document_processor.DocumentProcessor(doc_cfg, chunk_cfg)

        # Test markdown processing
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
            f.write("# Title\n\nContent here\n\n## Section\n\nMore content")
            md_path = Path(f.name)

        result = await processor.process_file(md_path)
        assert "content" in result
        assert "metadata" in result
        assert "chunks" in result
        md_path.unlink()

        # Test Python file processing
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def test():\n    return 42")
            py_path = Path(f.name)

        result = await processor.process_file(py_path)
        assert "content" in result
        py_path.unlink()

        # Test JSON processing
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump({"key": "value"}, f)
            json_path = Path(f.name)

        result = await processor.process_file(json_path)
        assert "content" in result
        json_path.unlink()

        # Test chunking
        long_text = " ".join(["word"] * 1000)
        chunks = processor._chunk_text(long_text)
        assert len(chunks) > 0

        # Test metadata extraction
        metadata = processor._extract_metadata(py_path)
        assert "file_type" in metadata

        # Test language detection
        lang = processor._detect_language(".py")
        assert lang == "python"

        lang = processor._detect_language(".js")
        assert lang == "javascript"

        lang = processor._detect_language(".unknown")
        assert lang is None


class TestEmbeddingsCoverage:
    """Achieve 80% coverage for embeddings.py"""

    @pytest.mark.asyncio
    async def test_embedding_providers(self):
        """Test different embedding providers"""
        # Test SentenceTransformer provider
        cfg = config.EmbeddingConfig(provider="sentence_transformers")

        with patch("eol.rag_context.embeddings.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode = MagicMock(return_value=np.array([[0.1] * 384]))
            MockST.return_value = mock_model

            manager = embeddings.EmbeddingManager(cfg)
            provider = manager._create_provider()

            # Test embed
            result = await provider.embed("test text")
            assert len(result) == 384

            # Test embed_batch
            results = await provider.embed_batch(["text1", "text2"])
            assert len(results) == 2

        # Test OpenAI provider
        cfg = config.EmbeddingConfig(provider="openai", api_key="test-key")

        with patch("eol.rag_context.embeddings.AsyncOpenAI") as MockOpenAI:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            MockOpenAI.return_value = mock_client

            manager = embeddings.EmbeddingManager(cfg)
            provider = manager._create_provider()

            result = await provider.embed("test text")
            assert len(result) == 1536

        # Test caching
        manager = embeddings.EmbeddingManager(cfg)
        manager.cache = {}

        # First call - cache miss
        with patch.object(manager.provider, "embed", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = np.array([0.1] * 384)
            result1 = await manager.get_embedding("test")
            assert mock_embed.called

        # Second call - cache hit
        with patch.object(manager.provider, "embed", new_callable=AsyncMock) as mock_embed:
            result2 = await manager.get_embedding("test")
            assert not mock_embed.called
            assert np.array_equal(result1, result2)


class TestIndexerCoverage:
    """Achieve 80% coverage for indexer.py"""

    @pytest.mark.asyncio
    async def test_document_indexing(self):
        """Test document indexing operations"""
        cfg = config.RAGConfig()

        with (
            patch("eol.rag_context.indexer.DocumentProcessor") as MockProcessor,
            patch("eol.rag_context.indexer.EmbeddingManager") as MockEmbedding,
            patch("eol.rag_context.indexer.RedisVectorStore") as MockRedis,
        ):

            mock_processor = AsyncMock()
            mock_processor.process_file = AsyncMock(
                return_value={
                    "content": "test content",
                    "chunks": [{"text": "chunk1"}, {"text": "chunk2"}],
                    "metadata": {"type": "python"},
                }
            )
            MockProcessor.return_value = mock_processor

            mock_embedding = AsyncMock()
            mock_embedding.get_embedding = AsyncMock(return_value=np.array([0.1] * 384))
            MockEmbedding.return_value = mock_embedding

            mock_redis = MagicMock()
            mock_redis.store_document = MagicMock(return_value="doc_id")
            MockRedis.return_value = mock_redis

            idx = indexer.DocumentIndexer(cfg, mock_processor, mock_embedding, mock_redis)

            # Test index_file
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
                f.write(b"def test(): pass")
                file_path = Path(f.name)

            result = await idx.index_file(file_path, "source1")
            assert result["status"] == "success"
            file_path.unlink()

            # Test index_directory
            with tempfile.TemporaryDirectory() as tmpdir:
                test_dir = Path(tmpdir)
                (test_dir / "test1.py").write_text("code1")
                (test_dir / "test2.md").write_text("# Doc")

                result = await idx.index_directory(
                    test_dir, recursive=True, patterns=["*.py", "*.md"]
                )
                assert "indexed_files" in result
                assert "total_chunks" in result

            # Test search
            mock_redis.search_similar_async = AsyncMock(
                return_value=[{"content": "result1", "score": 0.9}]
            )
            results = await idx.search("query", limit=5)
            assert len(results) > 0

    def test_folder_scanner(self):
        """Test FolderScanner class"""
        scanner = indexer.FolderScanner(
            patterns=["*.py", "*.md"], ignore_patterns=["*.pyc", "__pycache__"]
        )

        # Test should_process
        assert scanner.should_process(Path("test.py")) is True
        assert scanner.should_process(Path("test.pyc")) is False
        assert scanner.should_process(Path("other.txt")) is False

        # Test _should_ignore
        assert scanner._should_ignore(Path("__pycache__/file.py")) is True
        assert scanner._should_ignore(Path("normal.py")) is False


class TestSemanticCacheCoverage:
    """Achieve 80% coverage for semantic_cache.py"""

    @pytest.mark.asyncio
    async def test_semantic_caching(self):
        """Test semantic cache operations"""
        cache_cfg = config.SemanticCacheConfig(enabled=True)

        with (
            patch("eol.rag_context.semantic_cache.EmbeddingManager") as MockEmbedding,
            patch("eol.rag_context.semantic_cache.RedisVectorStore") as MockRedis,
        ):

            mock_embedding = AsyncMock()
            mock_embedding.get_embedding = AsyncMock(return_value=np.array([0.1] * 384))
            MockEmbedding.return_value = mock_embedding

            mock_redis = AsyncMock()
            mock_redis.redis = AsyncMock()
            mock_redis.redis.hset = AsyncMock()
            mock_redis.redis.expire = AsyncMock()
            mock_redis.redis.scan = AsyncMock(return_value=(0, ["key1", "key2"]))
            mock_redis.redis.hget = AsyncMock(return_value=json.dumps("cached_response"))
            mock_redis.search_similar_async = AsyncMock(
                return_value=[{"metadata": {"cache_key": "key1"}, "score": 0.95}]
            )
            MockRedis.return_value = mock_redis

            cache = semantic_cache.SemanticCache(cache_cfg, mock_embedding, mock_redis)
            await cache.initialize()

            # Test get - cache miss
            mock_redis.search_similar_async.return_value = []
            result = await cache.get("query1")
            assert result is None

            # Test set
            await cache.set("query1", "response1", {"meta": "data"})

            # Test get - cache hit
            mock_redis.search_similar_async.return_value = [
                {"metadata": {"cache_key": "key1"}, "score": 0.95}
            ]
            result = await cache.get("query1")
            assert result is not None

            # Test clear
            mock_redis.redis.delete = AsyncMock()
            await cache.clear()

            # Test stats
            stats = await cache.get_stats()
            assert "queries" in stats
            assert "hits" in stats


class TestKnowledgeGraphCoverage:
    """Achieve 80% coverage for knowledge_graph.py"""

    @pytest.mark.asyncio
    async def test_knowledge_graph_operations(self):
        """Test knowledge graph building"""
        kg_cfg = config.KnowledgeGraphConfig(enabled=True)

        with patch("eol.rag_context.knowledge_graph.RedisVectorStore") as MockRedis:
            mock_redis = MagicMock()
            mock_redis.redis = MagicMock()
            MockRedis.return_value = mock_redis

            builder = knowledge_graph.KnowledgeGraphBuilder(kg_cfg, mock_redis)

            # Test extract_entities
            text = "class MyClass: def method(self): pass"
            entities = builder.extract_entities(text, "python")
            assert len(entities) > 0

            # Test add_entity
            await builder.add_entity("TestClass", "class", {"file": "test.py"})

            # Test add_relationship
            await builder.add_relationship("ClassA", "inherits", "ClassB")

            # Test query_graph
            mock_redis.redis.smembers = MagicMock(return_value={"rel1", "rel2"})
            mock_redis.redis.hgetall = MagicMock(
                return_value={"source": "A", "type": "uses", "target": "B"}
            )

            result = await builder.query_graph("TestClass", max_depth=2)
            assert "entities" in result
            assert "relationships" in result


class TestFileWatcherCoverage:
    """Achieve 80% coverage for file_watcher.py"""

    @pytest.mark.asyncio
    async def test_file_watching(self):
        """Test file watcher operations"""
        watcher_cfg = config.FileWatcherConfig(enabled=True)

        with (
            patch("eol.rag_context.file_watcher.Observer") as MockObserver,
            patch("eol.rag_context.file_watcher.DocumentIndexer") as MockIndexer,
        ):

            mock_observer = MagicMock()
            MockObserver.return_value = mock_observer

            mock_indexer = AsyncMock()
            MockIndexer.return_value = mock_indexer

            watcher = file_watcher.FileWatcher(watcher_cfg, mock_indexer)

            # Test watch
            with tempfile.TemporaryDirectory() as tmpdir:
                watch_path = Path(tmpdir)
                watch_id = await watcher.watch(watch_path, ["*.py"], ["*.pyc"])
                assert watch_id is not None

                # Test is_watching
                assert watcher.is_watching(watch_id) is True

                # Test get_watch_info
                info = watcher.get_watch_info(watch_id)
                assert info is not None
                assert info["path"] == str(watch_path)

                # Test unwatch
                await watcher.unwatch(watch_id)
                assert watcher.is_watching(watch_id) is False

            # Test stop
            await watcher.stop()
            mock_observer.stop.assert_called()


class TestServerCoverage:
    """Achieve 80% coverage for server.py"""

    @pytest.mark.asyncio
    async def test_server_operations(self):
        """Test MCP server operations"""
        cfg = config.RAGConfig()

        with (
            patch("eol.rag_context.server.RedisVectorStore") as MockRedis,
            patch("eol.rag_context.server.DocumentIndexer") as MockIndexer,
            patch("eol.rag_context.server.SemanticCache") as MockCache,
            patch("eol.rag_context.server.KnowledgeGraphBuilder") as MockGraph,
            patch("eol.rag_context.server.FileWatcher") as MockWatcher,
            patch("eol.rag_context.server.FastMCP") as MockMCP,
        ):

            mock_redis = AsyncMock()
            MockRedis.return_value = mock_redis

            mock_indexer = AsyncMock()
            mock_indexer.index_directory = AsyncMock(
                return_value={"status": "success", "indexed_files": 5, "total_chunks": 20}
            )
            mock_indexer.search = AsyncMock(return_value=[{"content": "result", "score": 0.9}])
            MockIndexer.return_value = mock_indexer

            mock_cache = AsyncMock()
            mock_cache.initialize = AsyncMock()
            MockCache.return_value = mock_cache

            mock_graph = AsyncMock()
            MockGraph.return_value = mock_graph

            mock_watcher = AsyncMock()
            MockWatcher.return_value = mock_watcher

            mock_mcp = MagicMock()
            mock_mcp.tool = MagicMock()
            MockMCP.return_value = mock_mcp

            srv = server.EOLRAGContextServer(cfg)

            # Test initialize
            await srv.initialize()
            assert srv._initialized is True

            # Test index_directory
            result = await srv.index_directory("/test/path")
            assert result["status"] == "success"

            # Test search_context
            results = await srv.search_context("query", limit=5)
            assert len(results) > 0

            # Test get_stats
            stats = await srv.get_stats()
            assert "indexer" in stats

            # Test watch_directory
            result = await srv.watch_directory("/test/path")
            assert "watch_id" in result

            # Test clear_cache
            result = await srv.clear_cache()
            assert result["status"] == "success"


class TestMainCoverage:
    """Achieve 80% coverage for main.py"""

    def test_main_error_handling(self):
        """Test main function error paths"""
        # Test server run error
        with (
            patch("eol.rag_context.main.sys.argv", ["prog"]),
            patch("eol.rag_context.main.EOLRAGContextServer") as MockServer,
            patch("eol.rag_context.main.asyncio.run", side_effect=Exception("Server error")),
            patch("eol.rag_context.main.sys.exit") as mock_exit,
        ):

            mock_exit.side_effect = SystemExit
            try:
                main.main()
            except SystemExit:
                pass
            mock_exit.assert_called_with(1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=eol.rag_context", "--cov-report=term"])
