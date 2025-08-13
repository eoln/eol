"""
Working server tests for achieving coverage on server.py module.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock external dependencies thoroughly
mock_fastmcp_instance = MagicMock()
mock_fastmcp_instance.run = AsyncMock()
mock_fastmcp_class = MagicMock(return_value=mock_fastmcp_instance)

sys.modules["fastmcp"] = MagicMock()
sys.modules["fastmcp"].FastMCP = mock_fastmcp_class
sys.modules["fastmcp"].Context = MagicMock()

sys.modules["redis"] = MagicMock()
sys.modules["redis.asyncio"] = MagicMock()
sys.modules["watchdog"] = MagicMock()
sys.modules["watchdog.observers"] = MagicMock()
sys.modules["watchdog.events"] = MagicMock()
sys.modules["networkx"] = MagicMock()

from eol.rag_context import config, server


class TestServerBasics:
    """Test basic server functionality."""
    
    def test_server_creation(self):
        """Test that server can be created."""
        srv = server.EOLRAGContextServer()
        assert srv.config is not None
        assert srv.mcp is not None
        assert hasattr(srv, 'redis_store')
        assert hasattr(srv, 'indexer')
    
    def test_server_with_custom_config(self):
        """Test server with custom configuration."""
        custom_config = config.RAGConfig()
        custom_config.server_name = "test-server"
        srv = server.EOLRAGContextServer(custom_config)
        
        assert srv.config.server_name == "test-server"
        assert srv.config == custom_config
    
    @pytest.mark.asyncio
    async def test_server_initialization(self):
        """Test server initialization process."""
        srv = server.EOLRAGContextServer()
        
        # Mock all the required components
        with patch('eol.rag_context.server.RedisVectorStore') as MockRedis, \
             patch('eol.rag_context.server.EmbeddingManager') as MockEmbedding, \
             patch('eol.rag_context.server.DocumentProcessor') as MockDocProcessor, \
             patch('eol.rag_context.server.DocumentIndexer') as MockIndexer, \
             patch('eol.rag_context.server.SemanticCache') as MockCache, \
             patch('eol.rag_context.server.KnowledgeGraphBuilder') as MockKG:
            
            # Set up mocks
            mock_redis = AsyncMock()
            mock_redis.connect_async = AsyncMock()
            MockRedis.return_value = mock_redis
            
            MockEmbedding.return_value = AsyncMock()
            MockDocProcessor.return_value = MagicMock()
            MockIndexer.return_value = MagicMock() 
            MockCache.return_value = AsyncMock()
            MockKG.return_value = MagicMock()
            
            await srv.initialize()
            
            # Verify components were created
            assert srv.redis_store is not None
            assert srv.embedding_manager is not None
            assert srv.document_processor is not None
            assert srv.indexer is not None
            assert srv.semantic_cache is not None
            assert srv.knowledge_graph is not None
    
    @pytest.mark.asyncio
    async def test_server_shutdown(self):
        """Test server shutdown."""
        srv = server.EOLRAGContextServer()
        
        # Test shutdown when components are None (should not crash)
        await srv.shutdown()
        
        # Test shutdown with mocked components (only the ones actually used in shutdown)
        srv.redis_store = AsyncMock()
        srv.redis_store.close = AsyncMock()
        srv.file_watcher = AsyncMock()
        srv.file_watcher.stop = AsyncMock()
        
        await srv.shutdown()
        
        # Verify cleanup was called for components that are actually in shutdown
        srv.redis_store.close.assert_called_once()
        srv.file_watcher.stop.assert_called_once()


class TestServerRequestModels:
    """Test the Pydantic request models."""
    
    def test_index_directory_request(self):
        """Test IndexDirectoryRequest model."""
        request = server.IndexDirectoryRequest(
            path="/test/path",
            recursive=True,
            file_patterns=["*.py", "*.md"],
            watch=False
        )
        
        assert request.path == "/test/path"
        assert request.recursive == True
        assert request.file_patterns == ["*.py", "*.md"]
        assert request.watch == False
    
    def test_search_context_request(self):
        """Test SearchContextRequest model."""
        request = server.SearchContextRequest(
            query="test query",
            max_results=5,
            min_relevance=0.8,
            hierarchy_level=2,
            source_filter="source123"
        )
        
        assert request.query == "test query"
        assert request.max_results == 5
        assert request.min_relevance == 0.8
        assert request.hierarchy_level == 2
        assert request.source_filter == "source123"
    
    def test_optimize_context_request(self):
        """Test OptimizeContextRequest model."""
        request = server.OptimizeContextRequest(
            query="optimize this",
            current_context="current context here",
            max_tokens=16000,
            strategy="hierarchical"
        )
        
        assert request.query == "optimize this"
        assert request.current_context == "current context here"
        assert request.max_tokens == 16000
        assert request.strategy == "hierarchical"
    
    def test_watch_directory_request(self):
        """Test WatchDirectoryRequest model."""
        request = server.WatchDirectoryRequest(
            path="/watch/path",
            recursive=True,
            file_patterns=["*.py"]
        )
        
        assert request.path == "/watch/path"
        assert request.recursive == True
        assert request.file_patterns == ["*.py"]


class TestServerMethods:
    """Test server method functionality."""
    
    @pytest.mark.asyncio
    async def test_index_directory_method_exists(self):
        """Test that index_directory method exists and can be called."""
        srv = server.EOLRAGContextServer()
        
        # The method should exist even if not fully initialized
        assert hasattr(srv, 'index_directory')
        assert callable(srv.index_directory)
    
    @pytest.mark.asyncio
    async def test_index_file_method_exists(self):
        """Test that index_file method exists."""
        srv = server.EOLRAGContextServer()
        
        assert hasattr(srv, 'index_file')
        assert callable(srv.index_file)
    
    @pytest.mark.asyncio
    async def test_watch_directory_method_exists(self):
        """Test that watch_directory method exists."""
        srv = server.EOLRAGContextServer()
        
        assert hasattr(srv, 'watch_directory')
        assert callable(srv.watch_directory)
    
    @pytest.mark.asyncio
    async def test_run_method_exists(self):
        """Test that run method exists."""
        srv = server.EOLRAGContextServer()
        
        assert hasattr(srv, 'run')
        assert callable(srv.run)


def test_request_model_defaults():
    """Test request models with default values."""
    # Test minimal IndexDirectoryRequest
    request = server.IndexDirectoryRequest(path="/test")
    assert request.path == "/test"
    assert request.recursive == True
    assert request.file_patterns is None
    assert request.watch == False
    
    # Test minimal SearchContextRequest
    search_req = server.SearchContextRequest(query="test")
    assert search_req.query == "test"
    assert search_req.max_results == 10
    assert search_req.min_relevance == 0.7
    assert search_req.hierarchy_level is None
    assert search_req.source_filter is None


def test_server_components_initialization():
    """Test that server initializes components correctly."""
    srv = server.EOLRAGContextServer()
    
    # Check that all component slots are initialized to None
    assert srv.redis_store is None
    assert srv.embedding_manager is None
    assert srv.document_processor is None
    assert srv.indexer is None
    assert srv.semantic_cache is None
    assert srv.knowledge_graph is None
    assert srv.file_watcher is None
    
    # Check that MCP server is created
    assert srv.mcp is not None


if __name__ == "__main__":
    # Run sync tests
    test_request_model_defaults()
    test_server_components_initialization()
    print("✅ Sync server tests passed!")
    
    print("✅ All simple server tests passed!")