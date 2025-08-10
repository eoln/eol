"""
Unit tests for MCP server with proper mocking.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import numpy as np
import json
import tempfile

# Mock all external dependencies
for module in ['magic', 'pypdf', 'docx', 'redis', 'redis.asyncio', 'redis.commands',
               'redis.commands.search', 'redis.commands.search.field',
               'redis.commands.search.indexDefinition', 'redis.commands.search.query',
               'watchdog', 'watchdog.observers', 'watchdog.events', 'networkx',
               'sentence_transformers', 'openai', 'tree_sitter', 'yaml', 'bs4', 'aiofiles']:
    if module not in sys.modules:
        sys.modules[module] = MagicMock()

# Mock FastMCP before importing server
mock_fastmcp = MagicMock()
mock_server_class = MagicMock()
mock_fastmcp.FastMCP = mock_server_class
sys.modules['fastmcp'] = mock_fastmcp
sys.modules['fastmcp.server'] = MagicMock()

# Now import the server module
from eol.rag_context import server
from eol.rag_context.config import RAGConfig


class TestMCPServer:
    """Test MCP server functionality."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components."""
        components = MagicMock()
        components.config = RAGConfig()
        components.processor = MagicMock()
        components.embeddings = MagicMock()
        components.redis = MagicMock()
        components.indexer = MagicMock()
        components.cache = MagicMock()
        components.graph = MagicMock()
        components.watcher = MagicMock()
        return components
    
    @pytest.fixture
    def rag_server(self, mock_components):
        """Create RAG context server with mocks."""
        with patch('eol.rag_context.server.RAGComponents') as MockComponents:
            MockComponents.return_value = mock_components
            srv = server.RAGContextServer()
            srv.components = mock_components
            return srv
    
    def test_server_initialization(self):
        """Test server initialization."""
        with patch('eol.rag_context.server.RAGComponents') as MockComponents:
            mock_comp = MagicMock()
            MockComponents.return_value = mock_comp
            
            srv = server.RAGContextServer()
            assert srv.components is not None
            assert srv.mcp is not None
            mock_server_class.assert_called_with("eol-rag-context")
    
    @pytest.mark.asyncio
    async def test_initialize_server(self, rag_server):
        """Test server initialization."""
        # Mock component initialization
        rag_server.components.initialize = AsyncMock()
        
        await rag_server.initialize()
        
        rag_server.components.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_index_directory_tool(self, rag_server):
        """Test index_directory tool."""
        # Mock indexer
        rag_server.components.indexer.index_folder = AsyncMock(
            return_value=MagicMock(
                source_id="src123",
                file_count=10,
                total_chunks=50
            )
        )
        
        result = await rag_server.index_directory("/test/path")
        
        assert "src123" in result
        assert "10 files" in result
        assert "50 chunks" in result
    
    @pytest.mark.asyncio
    async def test_search_context_tool(self, rag_server):
        """Test search_context tool."""
        # Mock search results
        mock_results = [
            MagicMock(
                content="Result 1",
                metadata={"score": 0.9}
            ),
            MagicMock(
                content="Result 2",
                metadata={"score": 0.8}
            )
        ]
        rag_server.components.redis.search = AsyncMock(return_value=mock_results)
        
        result = await rag_server.search_context("test query", limit=2)
        
        assert "Result 1" in result
        assert "Result 2" in result
    
    @pytest.mark.asyncio
    async def test_query_knowledge_graph_tool(self, rag_server):
        """Test query_knowledge_graph tool."""
        # Mock graph query
        mock_subgraph = {
            "entities": [{"id": "e1", "name": "Entity 1"}],
            "relationships": [{"source": "e1", "target": "e2"}]
        }
        rag_server.components.graph.query_subgraph = AsyncMock(
            return_value=mock_subgraph
        )
        
        result = await rag_server.query_knowledge_graph("test entity", max_depth=2)
        
        assert "Entity 1" in result
        assert "entities" in result.lower()
    
    @pytest.mark.asyncio
    async def test_watch_directory_tool(self, rag_server):
        """Test watch_directory tool."""
        # Mock watcher
        rag_server.components.watcher.watch = AsyncMock(return_value="watch123")
        
        result = await rag_server.watch_directory("/test/path")
        
        assert "watch123" in result
        assert "watching" in result.lower()
    
    @pytest.mark.asyncio
    async def test_unwatch_directory_tool(self, rag_server):
        """Test unwatch_directory tool."""
        # Mock unwatcher
        rag_server.components.watcher.unwatch = AsyncMock(return_value=True)
        
        result = await rag_server.unwatch_directory("watch123")
        
        assert "unwatched" in result.lower()
    
    @pytest.mark.asyncio
    async def test_optimize_context_tool(self, rag_server):
        """Test optimize_context tool."""
        # Mock optimization
        mock_report = {
            "recommendations": ["Adjust threshold to 0.85"],
            "current_hit_rate": 0.25,
            "target_hit_rate": 0.31
        }
        rag_server.components.cache.get_optimization_report = AsyncMock(
            return_value=mock_report
        )
        
        result = await rag_server.optimize_context()
        
        assert "0.25" in result or "25" in result
        assert "recommendations" in result.lower()
    
    @pytest.mark.asyncio
    async def test_clear_cache_tool(self, rag_server):
        """Test clear_cache tool."""
        # Mock cache clear
        rag_server.components.cache.clear = AsyncMock()
        
        result = await rag_server.clear_cache()
        
        assert "cleared" in result.lower()
        rag_server.components.cache.clear.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_remove_source_tool(self, rag_server):
        """Test remove_source tool."""
        # Mock source removal
        rag_server.components.indexer.remove_source = AsyncMock(return_value=True)
        
        result = await rag_server.remove_source("src123")
        
        assert "removed" in result.lower()
        assert "src123" in result
    
    @pytest.mark.asyncio
    async def test_get_context_resource(self, rag_server):
        """Test get_context resource."""
        # Mock context retrieval
        mock_context = [
            MagicMock(content="Context 1", metadata={}),
            MagicMock(content="Context 2", metadata={})
        ]
        rag_server.components.redis.get_context = AsyncMock(return_value=mock_context)
        
        result = await rag_server.get_context("context://test-query")
        
        assert "Context 1" in result
        assert "Context 2" in result
    
    @pytest.mark.asyncio
    async def test_list_sources_resource(self, rag_server):
        """Test list_sources resource."""
        # Mock source listing
        mock_sources = [
            {"source_id": "src1", "path": "/path1"},
            {"source_id": "src2", "path": "/path2"}
        ]
        rag_server.components.indexer.list_sources = AsyncMock(
            return_value=mock_sources
        )
        
        result = await rag_server.list_sources()
        
        assert "src1" in result
        assert "src2" in result
    
    @pytest.mark.asyncio
    async def test_stats_resource(self, rag_server):
        """Test stats resource."""
        # Mock stats
        mock_stats = {
            "documents_indexed": 100,
            "cache_hit_rate": 0.31,
            "entities": 50
        }
        rag_server.components.indexer.get_stats = MagicMock(
            return_value={"documents_indexed": 100}
        )
        rag_server.components.cache.get_stats = MagicMock(
            return_value={"hit_rate": 0.31}
        )
        rag_server.components.graph.get_graph_stats = MagicMock(
            return_value={"entity_count": 50}
        )
        
        result = await rag_server.get_stats()
        
        assert "100" in result
        assert "0.31" in result or "31" in result
        assert "50" in result
    
    @pytest.mark.asyncio
    async def test_structured_query_prompt(self, rag_server):
        """Test structured_query prompt."""
        # Mock components for prompt
        rag_server.components.redis.search = AsyncMock(
            return_value=[MagicMock(content="Result", metadata={})]
        )
        
        result = await rag_server.structured_query("Find information about testing")
        
        assert len(result) > 0
        assert "structured" in result.lower() or "query" in result.lower()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, rag_server):
        """Test error handling in tools."""
        # Mock error
        rag_server.components.indexer.index_folder = AsyncMock(
            side_effect=Exception("Test error")
        )
        
        result = await rag_server.index_directory("/test/path")
        
        assert "error" in result.lower()
        assert "Test error" in result
    
    def test_tool_registration(self, rag_server):
        """Test that tools are registered with MCP."""
        # Check that mcp.tool was called for each tool
        assert rag_server.mcp.tool.called
        
        # Get all tool names registered
        tool_calls = [call[0][0] if call[0] else None 
                     for call in rag_server.mcp.tool.call_args_list]
        
        # Should have registered multiple tools
        assert len(tool_calls) > 0
    
    def test_resource_registration(self, rag_server):
        """Test that resources are registered with MCP."""
        # Check that mcp.resource was called
        assert rag_server.mcp.resource.called
        
        # Get all resource URIs registered
        resource_calls = [call[0][0] if call[0] else None 
                         for call in rag_server.mcp.resource.call_args_list]
        
        # Should have registered resources
        assert len(resource_calls) > 0
    
    def test_prompt_registration(self, rag_server):
        """Test that prompts are registered with MCP."""
        # Check that mcp.prompt was called
        assert rag_server.mcp.prompt.called


class TestRAGComponents:
    """Test RAG components initialization."""
    
    @pytest.mark.asyncio
    async def test_components_initialization(self):
        """Test RAGComponents initialization."""
        with patch('eol.rag_context.server.RedisVectorStore') as MockRedis, \
             patch('eol.rag_context.server.DocumentProcessor') as MockProcessor, \
             patch('eol.rag_context.server.EmbeddingManager') as MockEmbeddings, \
             patch('eol.rag_context.server.DocumentIndexer') as MockIndexer, \
             patch('eol.rag_context.server.SemanticCache') as MockCache, \
             patch('eol.rag_context.server.KnowledgeGraphBuilder') as MockGraph, \
             patch('eol.rag_context.server.FileWatcher') as MockWatcher:
            
            # Create mock instances
            mock_redis = MagicMock()
            mock_redis.connect_async = AsyncMock()
            mock_redis.create_indexes = AsyncMock()
            MockRedis.return_value = mock_redis
            
            # Initialize components
            components = server.RAGComponents()
            await components.initialize()
            
            # Verify initialization
            assert components.redis is not None
            assert components.processor is not None
            assert components.embeddings is not None
            assert components.indexer is not None
            assert components.cache is not None
            assert components.graph is not None
            assert components.watcher is not None
            
            # Verify async calls
            mock_redis.connect_async.assert_called_once()
            mock_redis.create_indexes.assert_called_once()
    
    def test_components_config(self):
        """Test RAGComponents configuration."""
        components = server.RAGComponents()
        
        assert components.config is not None
        assert hasattr(components.config, 'redis')
        assert hasattr(components.config, 'embedding')
        assert hasattr(components.config, 'chunking')


class TestServerHelpers:
    """Test server helper functions."""
    
    def test_format_error(self):
        """Test error formatting."""
        # Create a server instance
        with patch('eol.rag_context.server.RAGComponents'):
            srv = server.RAGContextServer()
            
            # Test error formatting
            error = Exception("Test error message")
            result = srv._format_error(error)
            
            assert "error" in result.lower()
            assert "Test error message" in result
    
    def test_format_results(self):
        """Test results formatting."""
        with patch('eol.rag_context.server.RAGComponents'):
            srv = server.RAGContextServer()
            
            # Test with list of results
            results = [
                MagicMock(content="Result 1", metadata={"score": 0.9}),
                MagicMock(content="Result 2", metadata={"score": 0.8})
            ]
            
            formatted = srv._format_results(results)
            
            assert "Result 1" in formatted
            assert "Result 2" in formatted
    
    def test_parse_uri(self):
        """Test URI parsing."""
        with patch('eol.rag_context.server.RAGComponents'):
            srv = server.RAGContextServer()
            
            # Test context URI
            uri = "context://test-query"
            parsed = srv._parse_uri(uri)
            
            assert parsed == "test-query"
            
            # Test source URI
            uri = "source://src123"
            parsed = srv._parse_uri(uri)
            
            assert parsed == "src123"


class TestServerRun:
    """Test server run method."""
    
    def test_run_method(self):
        """Test the run method."""
        with patch('eol.rag_context.server.RAGComponents'), \
             patch.object(server.RAGContextServer, 'mcp') as mock_mcp:
            
            mock_mcp_instance = MagicMock()
            mock_mcp.return_value = mock_mcp_instance
            
            srv = server.RAGContextServer()
            
            # Mock the run method
            mock_mcp_instance.run = MagicMock()
            
            # Call run
            srv.run()
            
            # Verify run was called
            mock_mcp_instance.run.assert_called_once()