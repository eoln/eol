"""
Tests for MCP server functionality.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from eol.rag_context.server import EOLRAGContextServer
from eol.rag_context.config import RAGConfig


class TestMCPServer:
    """Test MCP server functionality."""
    
    @pytest.fixture
    async def server(self, test_config):
        """Create MCP server instance."""
        server = EOLRAGContextServer(test_config)
        
        # Mock components to avoid Redis dependency
        server.redis_store = AsyncMock()
        server.embedding_manager = AsyncMock()
        server.document_processor = Mock()
        server.indexer = AsyncMock()
        server.semantic_cache = AsyncMock()
        server.knowledge_graph = AsyncMock()
        server.file_watcher = AsyncMock()
        
        # Mock methods
        server.indexer.index_folder = AsyncMock(return_value=Mock(
            source_id="test_source",
            path=Path("/test"),
            indexed_at=1234567890,
            file_count=10,
            total_chunks=50
        ))
        
        server.indexer.list_sources = AsyncMock(return_value=[])
        server.indexer.get_stats = Mock(return_value={"documents_indexed": 0})
        
        server.semantic_cache.get = AsyncMock(return_value=None)
        server.semantic_cache.set = AsyncMock()
        server.semantic_cache.get_stats = Mock(return_value={"hits": 0, "misses": 0})
        
        server.embedding_manager.get_embedding = AsyncMock(return_value=Mock())
        server.embedding_manager.get_cache_stats = Mock(return_value={"hits": 0})
        
        server.redis_store.hierarchical_search = AsyncMock(return_value=[
            {"id": "1", "content": "Test content", "score": 0.9}
        ])
        
        server.knowledge_graph.build_from_documents = AsyncMock()
        server.knowledge_graph.query_subgraph = AsyncMock(return_value=Mock(
            entities=[],
            relationships=[],
            central_entities=[],
            metadata={}
        ))
        server.knowledge_graph.get_graph_stats = Mock(return_value={"entity_count": 0})
        
        server.file_watcher.watch = AsyncMock(return_value="source_123")
        server.file_watcher.unwatch = AsyncMock(return_value=True)
        server.file_watcher.get_stats = Mock(return_value={"watched_sources": 0})
        
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
        # Get the tool function
        tools = server.mcp._tools
        index_tool = None
        for tool in tools.values():
            if tool.name == "index_directory":
                index_tool = tool
                break
        
        assert index_tool is not None
        
        # Test tool execution
        from eol.rag_context.server import IndexDirectoryRequest
        request = IndexDirectoryRequest(
            path="/test/path",
            recursive=True,
            watch=False
        )
        
        result = await index_tool.function(request, Mock())
        
        assert result["source_id"] == "test_source"
        assert result["file_count"] == 10
        assert result["total_chunks"] == 50
        server.indexer.index_folder.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_context_tool(self, server):
        """Test search_context MCP tool."""
        tools = server.mcp._tools
        search_tool = None
        for tool in tools.values():
            if tool.name == "search_context":
                search_tool = tool
                break
        
        assert search_tool is not None
        
        from eol.rag_context.server import SearchContextRequest
        request = SearchContextRequest(
            query="test query",
            max_results=5,
            min_relevance=0.7
        )
        
        result = await search_tool.function(request, Mock())
        
        assert len(result) > 0
        assert result[0]["score"] >= request.min_relevance
        server.embedding_manager.get_embedding.assert_called()
    
    @pytest.mark.asyncio
    async def test_query_knowledge_graph_tool(self, server):
        """Test query_knowledge_graph MCP tool."""
        tools = server.mcp._tools
        kg_tool = None
        for tool in tools.values():
            if tool.name == "query_knowledge_graph":
                kg_tool = tool
                break
        
        assert kg_tool is not None
        
        from eol.rag_context.server import QueryKnowledgeGraphRequest
        request = QueryKnowledgeGraphRequest(
            query="test query",
            max_depth=2,
            max_entities=10
        )
        
        result = await kg_tool.function(request, Mock())
        
        assert "entities" in result
        assert "relationships" in result
        assert result["query"] == "test query"
        server.knowledge_graph.query_subgraph.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_watch_directory_tool(self, server):
        """Test watch_directory MCP tool."""
        tools = server.mcp._tools
        watch_tool = None
        for tool in tools.values():
            if tool.name == "watch_directory":
                watch_tool = tool
                break
        
        assert watch_tool is not None
        
        from eol.rag_context.server import WatchDirectoryRequest
        request = WatchDirectoryRequest(
            path="/test/path",
            recursive=True
        )
        
        result = await watch_tool.function(request, Mock())
        
        assert result["source_id"] == "source_123"
        assert result["status"] == "watching"
        server.file_watcher.watch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_context_resource(self, server):
        """Test context retrieval resource."""
        resources = server.mcp._resources
        context_resource = None
        for resource in resources.values():
            if "query" in resource.uri:
                context_resource = resource
                break
        
        assert context_resource is not None
        
        # Test with no cache hit
        result = await context_resource.function("test query")
        
        assert result["query"] == "test query"
        assert "context" in result
        assert result["cached"] is False
        server.semantic_cache.get.assert_called()
        server.redis_store.hierarchical_search.assert_called()
    
    @pytest.mark.asyncio
    async def test_list_sources_resource(self, server):
        """Test list sources resource."""
        resources = server.mcp._resources
        sources_resource = None
        for resource in resources.values():
            if "sources" in resource.uri:
                sources_resource = resource
                break
        
        assert sources_resource is not None
        
        result = await sources_resource.function()
        
        assert isinstance(result, list)
        server.indexer.list_sources.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stats_resource(self, server):
        """Test statistics resource."""
        resources = server.mcp._resources
        stats_resource = None
        for resource in resources.values():
            if resource.uri == "context://stats":
                stats_resource = resource
                break
        
        assert stats_resource is not None
        
        result = await stats_resource.function()
        
        assert "indexer" in result
        assert "cache" in result
        assert "embeddings" in result
        assert "watcher" in result
        assert "knowledge_graph" in result
    
    @pytest.mark.asyncio
    async def test_structured_query_prompt(self, server):
        """Test structured query prompt."""
        prompts = server.mcp._prompts
        query_prompt = None
        for prompt in prompts.values():
            if prompt.name == "structured_query":
                query_prompt = prompt
                break
        
        assert query_prompt is not None
        
        result = await query_prompt.function()
        
        assert "Main Intent" in result
        assert "Key Entities" in result
        assert "Context Level" in result
    
    @pytest.mark.asyncio
    async def test_optimize_context_tool(self, server):
        """Test optimize_context MCP tool."""
        tools = server.mcp._tools
        optimize_tool = None
        for tool in tools.values():
            if tool.name == "optimize_context":
                optimize_tool = tool
                break
        
        assert optimize_tool is not None
        
        from eol.rag_context.server import OptimizeContextRequest
        request = OptimizeContextRequest(
            query="test query",
            max_tokens=1000,
            strategy="hierarchical"
        )
        
        result = await optimize_tool.function(request, Mock())
        
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
        with patch('eol.rag_context.server.RedisVectorStore') as MockRedis:
            mock_redis = AsyncMock()
            MockRedis.return_value = mock_redis
            mock_redis.connect_async = AsyncMock()
            mock_redis.close = AsyncMock()
            mock_redis.create_hierarchical_indexes = Mock()
            
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
        tools = server.mcp._tools
        clear_tool = None
        for tool in tools.values():
            if tool.name == "clear_cache":
                clear_tool = tool
                break
        
        assert clear_tool is not None
        
        result = await clear_tool.function(Mock())
        
        assert result["semantic_cache"] == "cleared"
        assert result["embedding_cache"] == "cleared"
        server.semantic_cache.clear.assert_called_once()
        server.embedding_manager.clear_cache.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_remove_source_tool(self, server):
        """Test remove_source MCP tool."""
        tools = server.mcp._tools
        remove_tool = None
        for tool in tools.values():
            if tool.name == "remove_source":
                remove_tool = tool
                break
        
        assert remove_tool is not None
        
        server.indexer.remove_source = AsyncMock(return_value=True)
        
        result = await remove_tool.function("source_123", Mock())
        
        assert result["source_id"] == "source_123"
        assert result["removed"] is True
        assert result["status"] == "removed"
        server.file_watcher.unwatch.assert_called_once_with("source_123")
        server.indexer.remove_source.assert_called_once_with("source_123")