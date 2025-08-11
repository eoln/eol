"""
Improved tests for server.py to boost coverage from 50% to 70%.
"""

import pytest
import sys
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
import json
import numpy as np

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
    "networkx",
    "fastmcp",
    "fastmcp.server",
    "aiofiles",
    "watchdog",
    "watchdog.observers",
    "watchdog.events",
    "pypdf",
    "docx",
    "magic",
    "tree_sitter",
    "yaml",
    "bs4",
    "markdown",
    "sentence_transformers",
    "openai",
    "gitignore_parser",
]:
    sys.modules[module] = MagicMock()

from eol.rag_context import config
from eol.rag_context import server
from eol.rag_context import document_processor
from eol.rag_context import redis_client
from eol.rag_context import knowledge_graph


@pytest.mark.asyncio
async def test_server_initialization():
    """Test EOLRAGContextServer initialization."""
    with patch("eol.rag_context.server.FastMCP") as MockMCP:
        mock_mcp = MagicMock()
        mock_mcp.tool = MagicMock()
        MockMCP.return_value = mock_mcp

        # Test with default config
        srv = server.EOLRAGContextServer()
        assert srv.config is not None
        assert srv.mcp is not None

        # Test with custom config
        custom_config = config.RAGConfig()
        srv = server.EOLRAGContextServer(custom_config)
        assert srv.config == custom_config


@pytest.mark.asyncio
async def test_server_initialize_method():
    """Test server initialize method."""
    with (
        patch("eol.rag_context.server.FastMCP") as MockMCP,
        patch("eol.rag_context.server.RedisVectorStore") as MockRedis,
        patch("eol.rag_context.server.EmbeddingManager") as MockEmb,
        patch("eol.rag_context.server.DocumentProcessor") as MockProc,
        patch("eol.rag_context.server.DocumentIndexer") as MockIdx,
        patch("eol.rag_context.server.SemanticCache") as MockCache,
        patch("eol.rag_context.server.KnowledgeGraphBuilder") as MockGraph,
        patch("eol.rag_context.server.FileWatcher") as MockWatcher,
    ):

        mock_mcp = MagicMock()
        MockMCP.return_value = mock_mcp

        mock_redis = MagicMock()
        mock_redis.connect_async = AsyncMock()
        mock_redis.create_hierarchical_indexes = MagicMock()
        MockRedis.return_value = mock_redis

        mock_emb = MagicMock()
        MockEmb.return_value = mock_emb

        srv = server.EOLRAGContextServer()
        await srv.initialize()

        assert srv.redis is not None
        assert srv.embeddings is not None
        assert srv.processor is not None
        assert srv.indexer is not None
        assert srv.cache is not None
        assert srv.graph is not None
        assert srv.watcher is not None

        # Test initialization error
        mock_redis.connect_async = AsyncMock(side_effect=Exception("Connection failed"))
        srv2 = server.EOLRAGContextServer()

        try:
            await srv2.initialize()
        except Exception as e:
            assert "Connection failed" in str(e)


@pytest.mark.asyncio
async def test_index_directory():
    """Test index_directory method."""
    srv = server.EOLRAGContextServer()

    # Mock components
    srv.indexer = MagicMock()
    srv.indexer.index_folder = AsyncMock(
        return_value=MagicMock(source_id="src123", file_count=10, total_chunks=50, errors=[])
    )
    srv.indexer.index_file = AsyncMock(return_value=MagicMock(source_id="src456", chunks=5))

    srv.watcher = MagicMock()
    srv.watcher.watch = AsyncMock(return_value="watch123")

    # Test indexing directory
    result = await srv.index_directory("/test/dir")
    assert result["status"] == "success"
    assert result["source_id"] == "src123"
    assert result["indexed_files"] == 10

    # Test indexing directory with watch
    result = await srv.index_directory("/test/dir", watch=True)
    assert "watch_id" in result

    # Test indexing file
    result = await srv.index_directory("/test/file.py")
    assert "indexed" in result

    # Test error handling
    srv.indexer.index_folder = AsyncMock(side_effect=Exception("Index error"))
    result = await srv.index_directory("/test/error")
    assert result["status"] == "error"
    assert "Index error" in result["error"]


@pytest.mark.asyncio
async def test_search_context():
    """Test search_context method."""
    srv = server.EOLRAGContextServer()

    # Mock redis
    srv.redis = MagicMock()

    # Create mock search results
    mock_results = []
    for i in range(3):
        mock_doc = MagicMock()
        mock_doc.content = f"Result {i+1} content"
        mock_doc.metadata = {"score": 0.9 - i * 0.1, "source": f"file{i}.py"}
        mock_results.append(mock_doc)

    srv.redis.vector_search = AsyncMock(return_value=mock_results)
    srv.redis.hierarchical_search = AsyncMock(return_value=mock_results)

    # Test basic search
    results = await srv.search_context("test query")
    assert len(results) == 3
    assert results[0]["content"] == "Result 1 content"

    # Test search with limit
    results = await srv.search_context("test query", limit=2)
    assert len(results) == 2

    # Test hierarchical search
    results = await srv.search_context("test query", hierarchy_level=2)
    assert len(results) == 3

    # Test error handling
    srv.redis.vector_search = AsyncMock(side_effect=Exception("Search error"))
    results = await srv.search_context("error query")
    assert results == []


@pytest.mark.asyncio
async def test_query_knowledge_graph():
    """Test query_knowledge_graph method."""
    srv = server.EOLRAGContextServer()

    # Mock graph
    srv.graph = MagicMock()
    srv.graph.query_subgraph = AsyncMock(
        return_value={
            "entities": [
                {"id": "e1", "name": "Entity1", "type": "class"},
                {"id": "e2", "name": "Entity2", "type": "function"},
            ],
            "relationships": [{"source": "e1", "target": "e2", "type": "uses"}],
        }
    )

    # Test query
    result = await srv.query_knowledge_graph("Entity1")
    assert len(result["entities"]) == 2
    assert len(result["relationships"]) == 1

    # Test with max_depth
    result = await srv.query_knowledge_graph("Entity1", max_depth=5)
    assert "entities" in result

    # Test error handling
    srv.graph.query_subgraph = AsyncMock(side_effect=Exception("Graph error"))
    result = await srv.query_knowledge_graph("error")
    assert result["entities"] == []
    assert result["relationships"] == []


@pytest.mark.asyncio
async def test_watch_and_unwatch_directory():
    """Test watch_directory and unwatch_directory methods."""
    srv = server.EOLRAGContextServer()

    # Mock watcher
    srv.watcher = MagicMock()
    srv.watcher.watch = AsyncMock(return_value="watch123")
    srv.watcher.unwatch = AsyncMock(return_value=True)

    # Test watch
    result = await srv.watch_directory("/test/dir")
    assert result["status"] == "success"
    assert result["watch_id"] == "watch123"

    # Test watch with patterns
    result = await srv.watch_directory("/test/dir", patterns=["*.py", "*.md"])
    assert result["status"] == "success"

    # Test watch with ignore patterns
    result = await srv.watch_directory("/test/dir", ignore=["*.pyc", "__pycache__"])
    assert result["status"] == "success"

    # Test unwatch
    result = await srv.unwatch_directory("watch123")
    assert result["status"] == "success"

    # Test unwatch non-existent
    srv.watcher.unwatch = AsyncMock(return_value=False)
    result = await srv.unwatch_directory("nonexistent")
    assert result["status"] == "error"

    # Test error handling
    srv.watcher.watch = AsyncMock(side_effect=Exception("Watch error"))
    result = await srv.watch_directory("/error")
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_optimize_context():
    """Test optimize_context method."""
    srv = server.EOLRAGContextServer()

    # Mock cache
    srv.cache = MagicMock()
    srv.cache.get_optimization_report = AsyncMock(
        return_value={
            "current_hit_rate": 0.28,
            "target_hit_rate": 0.31,
            "recommendations": [
                "Increase similarity threshold to 0.95",
                "Enable adaptive threshold",
            ],
            "cache_size": 500,
            "total_queries": 1000,
        }
    )

    # Test optimization
    result = await srv.optimize_context()
    assert "current_hit_rate" in result
    assert len(result["recommendations"]) > 0

    # Test with custom target
    result = await srv.optimize_context(target_hit_rate=0.35)
    assert "recommendations" in result

    # Test error handling
    srv.cache.get_optimization_report = AsyncMock(side_effect=Exception("Cache error"))
    result = await srv.optimize_context()
    assert result["recommendations"] == []


@pytest.mark.asyncio
async def test_clear_cache():
    """Test clear_cache method."""
    srv = server.EOLRAGContextServer()

    # Mock cache
    srv.cache = MagicMock()
    srv.cache.clear = AsyncMock()
    srv.cache.get_stats = MagicMock(return_value={"queries": 0, "hits": 0, "hit_rate": 0.0})

    # Test clear
    result = await srv.clear_cache()
    assert result["status"] == "success"
    assert result["cache_stats"]["queries"] == 0

    # Test error handling
    srv.cache.clear = AsyncMock(side_effect=Exception("Clear error"))
    result = await srv.clear_cache()
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_remove_source():
    """Test remove_source method."""
    srv = server.EOLRAGContextServer()

    # Mock indexer
    srv.indexer = MagicMock()
    srv.indexer.remove_source = AsyncMock(return_value=True)

    # Test successful removal
    result = await srv.remove_source("src123")
    assert result["status"] == "success"

    # Test non-existent source
    srv.indexer.remove_source = AsyncMock(return_value=False)
    result = await srv.remove_source("nonexistent")
    assert result["status"] == "error"

    # Test error handling
    srv.indexer.remove_source = AsyncMock(side_effect=Exception("Remove error"))
    result = await srv.remove_source("error")
    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_get_context():
    """Test get_context method."""
    srv = server.EOLRAGContextServer()

    # Mock redis
    srv.redis = MagicMock()

    mock_docs = []
    for i in range(5):
        mock_doc = MagicMock()
        mock_doc.content = f"Context chunk {i+1}"
        mock_doc.metadata = {"chunk_index": i}
        mock_docs.append(mock_doc)

    srv.redis.get_context = AsyncMock(return_value=mock_docs)

    # Test get context
    docs = await srv.get_context("context://test query")
    assert len(docs) == 5
    assert docs[0]["content"] == "Context chunk 1"

    # Test with invalid URI
    docs = await srv.get_context("invalid://query")
    assert docs == []

    # Test error handling
    srv.redis.get_context = AsyncMock(side_effect=Exception("Context error"))
    docs = await srv.get_context("context://error")
    assert docs == []


@pytest.mark.asyncio
async def test_list_sources():
    """Test list_sources method."""
    srv = server.EOLRAGContextServer()

    # Mock indexer
    srv.indexer = MagicMock()
    srv.indexer.list_sources = AsyncMock(
        return_value=[
            {"source_id": "src1", "path": "/test/dir1", "file_count": 10},
            {"source_id": "src2", "path": "/test/dir2", "file_count": 20},
        ]
    )

    # Test list sources
    sources = await srv.list_sources()
    assert len(sources) == 2
    assert sources[0]["source_id"] == "src1"

    # Test error handling
    srv.indexer.list_sources = AsyncMock(side_effect=Exception("List error"))
    sources = await srv.list_sources()
    assert sources == []


@pytest.mark.asyncio
async def test_get_stats():
    """Test get_stats method."""
    srv = server.EOLRAGContextServer()

    # Mock components
    srv.indexer = MagicMock()
    srv.indexer.get_stats = MagicMock(return_value={"total_documents": 100, "total_chunks": 500})

    srv.cache = MagicMock()
    srv.cache.get_stats = MagicMock(return_value={"queries": 1000, "hits": 310, "hit_rate": 0.31})

    srv.graph = MagicMock()
    srv.graph.get_graph_stats = MagicMock(return_value={"nodes": 200, "edges": 150})

    # Test get stats
    stats = await srv.get_stats()
    assert "indexer" in stats
    assert "cache" in stats
    assert "graph" in stats
    assert stats["indexer"]["total_documents"] == 100

    # Test error handling
    srv.indexer.get_stats = MagicMock(side_effect=Exception("Stats error"))
    stats = await srv.get_stats()
    assert stats["indexer"] == {}


@pytest.mark.asyncio
async def test_structured_query():
    """Test structured_query method."""
    srv = server.EOLRAGContextServer()

    # Mock redis
    srv.redis = MagicMock()

    mock_results = [
        MagicMock(content="Result 1", metadata={"type": "code"}),
        MagicMock(content="Result 2", metadata={"type": "doc"}),
    ]

    srv.redis.vector_search = AsyncMock(return_value=mock_results)

    # Test structured query
    result = await srv.structured_query("test query")
    assert len(result["results"]) == 2
    assert result["metadata"]["total"] == 2

    # Test with filters
    result = await srv.structured_query(
        "test query", filters={"type": "code", "language": "python"}
    )
    assert "results" in result

    # Test with options
    result = await srv.structured_query("test query", options={"boost": 2.0, "rerank": True})
    assert "results" in result

    # Test error handling
    srv.redis.vector_search = AsyncMock(side_effect=Exception("Query error"))
    result = await srv.structured_query("error")
    assert result["results"] == []


@pytest.mark.asyncio
async def test_run_method():
    """Test run method."""
    with patch("eol.rag_context.server.FastMCP") as MockMCP:
        mock_mcp = MagicMock()
        mock_mcp.run = AsyncMock()
        MockMCP.return_value = mock_mcp

        srv = server.EOLRAGContextServer()

        # Mock initialize
        srv.initialize = AsyncMock()

        # Test run
        await srv.run()
        assert mock_mcp.run.called


@pytest.mark.asyncio
async def test_request_models():
    """Test request model classes."""
    # Test IndexDirectoryRequest
    req = server.IndexDirectoryRequest(
        path="/test/dir", watch=True, ignore_patterns=["*.pyc", "__pycache__"]
    )
    assert req.path == "/test/dir"
    assert req.watch == True
    assert len(req.ignore_patterns) == 2

    # Test SearchContextRequest
    req = server.SearchContextRequest(
        query="test query", limit=20, hierarchy_level=2, filters={"type": "code"}
    )
    assert req.query == "test query"
    assert req.limit == 20

    # Test QueryKnowledgeGraphRequest
    req = server.QueryKnowledgeGraphRequest(entity="TestEntity", max_depth=5)
    assert req.entity == "TestEntity"
    assert req.max_depth == 5

    # Test OptimizeContextRequest
    req = server.OptimizeContextRequest(target_hit_rate=0.35, max_cache_size=2000)
    assert req.target_hit_rate == 0.35

    # Test WatchDirectoryRequest
    req = server.WatchDirectoryRequest(
        path="/test", patterns=["*.py"], ignore=["*.pyc"], recursive=True
    )
    assert req.path == "/test"
    assert req.recursive == True
