"""
Tests for MCP server functionality.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp import FastMCP

from eol.rag_context.config import RAGConfig
from eol.rag_context.server import EOLRAGContextServer


class TestMCPServer:
    """Test MCP server functionality."""

    @pytest.fixture
    async def server(self, test_config):
        """Create MCP server instance."""
        # Create server but prevent auto-registration
        server = object.__new__(EOLRAGContextServer)
        server.config = test_config
        server.mcp = FastMCP(name=test_config.server_name, version=test_config.server_version)

        # Mock components to avoid Redis dependency
        server.redis_store = AsyncMock()
        server.embedding_manager = AsyncMock()
        server.document_processor = Mock()
        server.indexer = AsyncMock()
        server.semantic_cache = AsyncMock()
        server.knowledge_graph = AsyncMock()
        server.file_watcher = AsyncMock()

        # Mock methods
        server.indexer.index_folder = AsyncMock(
            return_value=Mock(
                source_id="test_source",
                path=Path("/test"),
                indexed_at=1234567890,
                file_count=10,
                total_chunks=50,
            )
        )

        server.indexer.list_sources = AsyncMock(return_value=[])
        server.indexer.get_stats = Mock(return_value={"documents_indexed": 0})

        server.semantic_cache.get = AsyncMock(return_value=None)
        server.semantic_cache.set = AsyncMock()
        server.semantic_cache.get_stats = Mock(return_value={"hits": 0, "misses": 0})

        server.embedding_manager.get_embedding = AsyncMock(return_value=Mock())
        server.embedding_manager.get_cache_stats = Mock(return_value={"hits": 0})

        server.redis_store.hierarchical_search = AsyncMock(
            return_value=[{"id": "1", "content": "Test content", "score": 0.9}]
        )

        server.knowledge_graph.build_from_documents = AsyncMock()
        server.knowledge_graph.query_subgraph = AsyncMock(
            return_value=Mock(entities=[], relationships=[], central_entities=[], metadata={})
        )
        server.knowledge_graph.get_graph_stats = Mock(return_value={"entity_count": 0})

        server.file_watcher.watch = AsyncMock(return_value="source_123")
        server.file_watcher.unwatch = AsyncMock(return_value=True)
        server.file_watcher.get_stats = Mock(return_value={"watched_sources": 0})

        # Now register tools/resources/prompts with mocked components in place
        server._setup_resources()
        server._setup_tools()
        server._setup_prompts()

        return server

    @pytest.mark.asyncio
    async def test_server_initialization(self, test_config):
        """Test server initialization."""
        server = EOLRAGContextServer(test_config)

        assert server.config == test_config
        assert server.mcp is not None
        assert server.mcp.name == test_config.server_name

    @pytest.mark.asyncio
    async def test_index_directory_tool(self, server):
        """Test index_directory MCP tool."""
        # Get the tool function using public API
        tools = await server.mcp.get_tools()
        index_tool = None
        for tool_name, tool_func in tools.items():
            if tool_name == "index_directory":
                index_tool = tool_func
                break

        assert index_tool is not None

        # Test tool execution
        from eol.rag_context.server import IndexDirectoryRequest

        request = IndexDirectoryRequest(path="/test/path", recursive=True, watch=False)

        # FunctionTool objects have a run method that needs context
        from fastmcp.server.context import Context, _current_context

        _current_context.set(Context(fastmcp=server.mcp))
        result = await index_tool.run(request)

        assert result["source_id"] == "test_source"
        assert result["file_count"] == 10
        assert result["total_chunks"] == 50
        server.indexer.index_folder.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_context_tool(self, server):
        """Test search_context MCP tool."""
        tools = await server.mcp.get_tools()
        search_tool = None
        for tool_name, tool_func in tools.items():
            if tool_name == "search_context":
                search_tool = tool_func
                break

        assert search_tool is not None

        from eol.rag_context.server import SearchContextRequest

        request = SearchContextRequest(query="test query", max_results=5, min_relevance=0.7)

        from fastmcp.server.context import Context, _current_context

        _current_context.set(Context(fastmcp=server.mcp))
        result = await search_tool.run(request)

        assert len(result) > 0
        assert result[0]["score"] >= request.min_relevance
        server.embedding_manager.get_embedding.assert_called()

    @pytest.mark.asyncio
    async def test_query_knowledge_graph_tool(self, server):
        """Test query_knowledge_graph MCP tool."""
        tools = await server.mcp.get_tools()
        kg_tool = None
        for tool_name, tool_func in tools.items():
            if tool_name == "query_knowledge_graph":
                kg_tool = tool_func
                break

        assert kg_tool is not None

        from eol.rag_context.server import QueryKnowledgeGraphRequest

        request = QueryKnowledgeGraphRequest(query="test query", max_depth=2, max_entities=10)

        from fastmcp.server.context import Context, _current_context

        _current_context.set(Context(fastmcp=server.mcp))
        result = await kg_tool.run(request)

        assert "entities" in result
        assert "relationships" in result
        assert result["query"] == "test query"
        server.knowledge_graph.query_subgraph.assert_called_once()

    @pytest.mark.asyncio
    async def test_watch_directory_tool(self, server):
        """Test watch_directory MCP tool."""
        tools = await server.mcp.get_tools()
        watch_tool = None
        for tool_name, tool_func in tools.items():
            if tool_name == "watch_directory":
                watch_tool = tool_func
                break

        assert watch_tool is not None

        from eol.rag_context.server import WatchDirectoryRequest

        request = WatchDirectoryRequest(path="/test/path", recursive=True)

        from fastmcp.server.context import Context, _current_context

        _current_context.set(Context(fastmcp=server.mcp))
        result = await watch_tool.run(request)

        assert result["source_id"] == "source_123"
        assert result["status"] == "watching"
        server.file_watcher.watch.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_context_resource(self, server):
        """Test context retrieval resource."""
        resources = await server.mcp.get_resources()
        context_resource = None
        for resource_uri, resource_func in resources.items():
            if "query" in resource_uri:
                context_resource = resource_func
                break

        assert context_resource is not None

        # Test with no cache hit
        # Resources are FunctionResource objects with read method
        from fastmcp.server.context import Context, _current_context

        _current_context.set(Context(fastmcp=server.mcp))
        # Resources take a uri parameter
        result = await context_resource.read("context://query/test query")

        assert result["query"] == "test query"
        assert "context" in result
        assert result["cached"] is False
        server.semantic_cache.get.assert_called()
        server.redis_store.hierarchical_search.assert_called()

    @pytest.mark.asyncio
    async def test_list_sources_resource(self, server):
        """Test list sources resource."""
        resources = await server.mcp.get_resources()
        sources_resource = None
        for resource_uri, resource_func in resources.items():
            if "sources" in resource_uri:
                sources_resource = resource_func
                break

        assert sources_resource is not None

        from fastmcp.server.context import Context, _current_context

        _current_context.set(Context(fastmcp=server.mcp))
        result = await sources_resource.read("context://sources")

        assert isinstance(result, list)
        server.indexer.list_sources.assert_called_once()

    @pytest.mark.asyncio
    async def test_stats_resource(self, server):
        """Test statistics resource."""
        resources = await server.mcp.get_resources()
        stats_resource = None
        for resource_uri, resource_func in resources.items():
            if resource_uri == "context://stats":
                stats_resource = resource_func
                break

        assert stats_resource is not None

        from fastmcp.server.context import Context, _current_context

        _current_context.set(Context(fastmcp=server.mcp))
        result = await stats_resource.read("context://stats")

        assert "indexer" in result
        assert "cache" in result
        assert "embeddings" in result
        assert "watcher" in result
        assert "knowledge_graph" in result

    @pytest.mark.asyncio
    async def test_structured_query_prompt(self, server):
        """Test structured query prompt."""
        prompts = await server.mcp.get_prompts()
        query_prompt = None
        for prompt_name, prompt_func in prompts.items():
            if prompt_name == "structured_query":
                query_prompt = prompt_func
                break

        assert query_prompt is not None

        from fastmcp.server.context import Context, _current_context

        _current_context.set(Context(fastmcp=server.mcp))
        # Prompts use render method
        result = await query_prompt.render({})

        assert "Main Intent" in result
        assert "Key Entities" in result
        assert "Context Level" in result

    @pytest.mark.asyncio
    async def test_optimize_context_tool(self, server):
        """Test optimize_context MCP tool."""
        tools = await server.mcp.get_tools()
        optimize_tool = None
        for tool_name, tool_func in tools.items():
            if tool_name == "optimize_context":
                optimize_tool = tool_func
                break

        assert optimize_tool is not None

        from eol.rag_context.server import OptimizeContextRequest

        request = OptimizeContextRequest(
            query="test query", max_tokens=1000, strategy="hierarchical"
        )

        from fastmcp.server.context import Context, _current_context

        _current_context.set(Context(fastmcp=server.mcp))
        result = await optimize_tool.run(request)

        assert "optimized_context" in result
        assert result["query"] == "test query"
        assert result["strategy"] == "hierarchical"
        assert "estimated_tokens" in result


class TestMCPServerIntegration:
    """Integration tests for MCP server with mocked Redis."""

    @pytest.mark.asyncio
    async def test_server_lifecycle(self, test_config):
        """Test server initialization and shutdown."""
        server = EOLRAGContextServer(test_config)

        # Mock Redis connection
        with patch("eol.rag_context.server.RedisVectorStore") as MockRedis:
            mock_redis = AsyncMock()
            MockRedis.return_value = mock_redis
            mock_redis.connect_async = AsyncMock()
            mock_redis.close = AsyncMock()
            mock_redis.create_hierarchical_indexes = Mock()

            # Mock async_redis with proper ft() method
            mock_redis.async_redis = AsyncMock()
            mock_ft = AsyncMock()
            mock_ft.info = AsyncMock(side_effect=Exception("Index not found"))
            mock_ft.create_index = AsyncMock()
            mock_redis.async_redis.ft = Mock(return_value=mock_ft)
            mock_redis.async_redis.scan = AsyncMock(return_value=(0, []))
            mock_redis.async_redis.delete = AsyncMock()

            # Mock redis (sync) for semantic cache
            mock_redis.redis = Mock()
            mock_redis.redis.hgetall = AsyncMock(return_value={})
            mock_redis.redis.hset = AsyncMock()
            mock_redis.redis.expire = AsyncMock()
            mock_redis.redis.scan = AsyncMock(return_value=(0, []))
            mock_redis.redis.delete = AsyncMock()
            mock_redis.redis.hincrby = AsyncMock()
            mock_redis.redis.hget = AsyncMock(return_value=None)
            mock_ft_sync = AsyncMock()
            mock_ft_sync.search = AsyncMock(return_value=Mock(docs=[]))
            mock_redis.redis.ft = Mock(return_value=mock_ft_sync)

            # Initialize
            await server.initialize()

            assert server.redis_store is not None
            assert server.embedding_manager is not None
            assert server.indexer is not None

            # Shutdown
            await server.shutdown()

            mock_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_cache_tool(self, server):
        """Test clear_cache MCP tool."""
        tools = await server.mcp.get_tools()
        clear_tool = None
        for tool_name, tool_func in tools.items():
            if tool_name == "clear_cache":
                clear_tool = tool_func
                break

        assert clear_tool is not None

        # For tools with no parameters
        from fastmcp.server.context import Context, _current_context

        _current_context.set(Context(fastmcp=server.mcp))
        result = await clear_tool.run({})

        assert result["semantic_cache"] == "cleared"
        assert result["embedding_cache"] == "cleared"
        server.semantic_cache.clear.assert_called_once()
        server.embedding_manager.clear_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_source_tool(self, server):
        """Test remove_source MCP tool."""
        tools = await server.mcp.get_tools()
        remove_tool = None
        for tool_name, tool_func in tools.items():
            if tool_name == "remove_source":
                remove_tool = tool_func
                break

        assert remove_tool is not None

        # Already mocked in conftest, just ensure it returns True
        server.indexer.remove_source.return_value = True

        from fastmcp.server.context import Context, _current_context

        _current_context.set(Context(fastmcp=server.mcp))
        # Pass source_id as a proper argument dict
        result = await remove_tool.run({"source_id": "source_123"})

        assert result["source_id"] == "source_123"
        assert result["removed"] is True
        assert result["status"] == "removed"
        server.file_watcher.unwatch.assert_called_once_with("source_123")
        server.indexer.remove_source.assert_called_once_with("source_123")
