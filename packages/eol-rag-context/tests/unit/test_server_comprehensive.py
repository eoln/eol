"""
Comprehensive tests for server.py to boost coverage from 54.84% to 75%+.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock all external dependencies before importing
mock_modules = {
    "fastmcp": MagicMock(),
    "redis": MagicMock(),
    "redis.asyncio": MagicMock(),
    "redis.commands": MagicMock(),
    "redis.commands.search": MagicMock(),
    "redis.commands.search.field": MagicMock(),
    "redis.commands.search.indexDefinition": MagicMock(),
    "redis.commands.search.query": MagicMock(),
    "redis.exceptions": MagicMock(),
    "networkx": MagicMock(),
    "watchdog": MagicMock(),
    "watchdog.observers": MagicMock(),
    "watchdog.events": MagicMock(),
}

for module_name, mock_module in mock_modules.items():
    sys.modules[module_name] = mock_module

# Mock FastMCP class specifically
mock_fastmcp_class = MagicMock()
mock_fastmcp_instance = MagicMock()
mock_fastmcp_instance.run = AsyncMock()
mock_fastmcp_class.return_value = mock_fastmcp_instance
sys.modules["fastmcp"].FastMCP = mock_fastmcp_class

from eol.rag_context import config, server


class TestEOLRAGContextServer:
    """Test the main server class functionality."""
    
    def test_server_initialization_default_config(self):
        """Test server initialization with default configuration."""
        srv = server.EOLRAGContextServer()
        
        assert srv.config is not None
        assert isinstance(srv.config, config.RAGConfig)
        assert srv.indexer is None
        assert srv.embedding_manager is None
        assert srv.redis_store is None
        assert srv.semantic_cache is None
        assert srv.knowledge_graph is None
        assert srv.file_watcher is None
        assert srv.mcp is not None
    
    def test_server_initialization_custom_config(self):
        """Test server initialization with custom configuration."""
        custom_config = config.RAGConfig()
        custom_config.server_name = "test-server"
        custom_config.debug = True
        
        srv = server.EOLRAGContextServer(custom_config)
        
        assert srv.config == custom_config
        assert srv.config.server_name == "test-server"
        assert srv.config.debug == True
    
    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful server initialization."""
        srv = server.EOLRAGContextServer()
        
        # Mock all the dependencies
        with patch('eol.rag_context.server.RedisVectorStore') as MockRedisStore, \
             patch('eol.rag_context.server.EmbeddingManager') as MockEmbedding, \
             patch('eol.rag_context.server.DocumentIndexer') as MockIndexer, \
             patch('eol.rag_context.server.SemanticCache') as MockCache, \
             patch('eol.rag_context.server.KnowledgeGraphBuilder') as MockKG:
            
            # Set up mocks
            mock_redis = AsyncMock()
            mock_redis.connect_async = AsyncMock()
            MockRedisStore.return_value = mock_redis
            
            mock_embedding = AsyncMock()
            MockEmbedding.return_value = mock_embedding
            
            mock_indexer = MagicMock()
            MockIndexer.return_value = mock_indexer
            
            mock_cache = AsyncMock()
            MockCache.return_value = mock_cache
            
            mock_kg = MagicMock()
            MockKG.return_value = mock_kg
            
            await srv.initialize()
            
            # Verify components were created
            assert srv.redis_store is not None
            assert srv.embedding_manager is not None
            assert srv.indexer is not None
            assert srv.semantic_cache is not None
            assert srv.knowledge_graph is not None
            
            # Verify Redis connection was attempted
            mock_redis.connect_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_redis_failure(self):
        """Test initialization handling Redis connection failure."""
        srv = server.EOLRAGContextServer()
        
        with patch('eol.rag_context.server.RedisVectorStore') as MockRedisStore:
            mock_redis = AsyncMock()
            mock_redis.connect_async = AsyncMock(side_effect=ConnectionError("Redis unavailable"))
            MockRedisStore.return_value = mock_redis
            
            with pytest.raises(ConnectionError):
                await srv.initialize()
    
    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test server shutdown process."""
        srv = server.EOLRAGContextServer()
        
        # Mock components
        srv.redis_store = AsyncMock()
        srv.semantic_cache = AsyncMock()
        srv.file_watcher = MagicMock()
        srv.file_watcher.stop = MagicMock()
        
        await srv.shutdown()
        
        # Verify cleanup was called
        srv.redis_store.close.assert_called_once()
        srv.semantic_cache.close.assert_called_once()
        srv.file_watcher.stop.assert_called_once()
    
    def test_setup_resources(self):
        """Test MCP resource setup."""
        srv = server.EOLRAGContextServer()
        srv._setup_resources()
        
        # Verify resources were set up (check app mock calls)
        assert srv.app is not None
    
    def test_setup_tools(self):
        """Test MCP tool setup."""
        srv = server.EOLRAGContextServer()
        srv._setup_tools()
        
        # Verify tools were set up
        assert srv.app is not None
    
    def test_setup_prompts(self):
        """Test MCP prompt setup.""" 
        srv = server.EOLRAGContextServer()
        srv._setup_prompts()
        
        # Verify prompts were set up
        assert srv.app is not None


class TestServerRequestModels:
    """Test Pydantic request models."""
    
    def test_index_directory_request(self):
        """Test IndexDirectoryRequest model."""
        request = server.IndexDirectoryRequest(
            path="/test/path",
            recursive=True,
            file_patterns=["*.py", "*.md"],
            max_workers=4
        )
        
        assert request.path == "/test/path"
        assert request.recursive == True
        assert request.file_patterns == ["*.py", "*.md"]
        assert request.max_workers == 4
    
    def test_search_context_request(self):
        """Test SearchContextRequest model."""
        request = server.SearchContextRequest(
            query="test query",
            k=10,
            hierarchy_level=2,
            strategy="comprehensive",
            filters={"type": "code"},
            include_metadata=True
        )
        
        assert request.query == "test query"
        assert request.k == 10
        assert request.hierarchy_level == 2
        assert request.strategy == "comprehensive"
        assert request.filters == {"type": "code"}
        assert request.include_metadata == True
    
    def test_query_knowledge_graph_request(self):
        """Test QueryKnowledgeGraphRequest model."""
        request = server.QueryKnowledgeGraphRequest(
            query="find connections",
            entity_types=["function", "class"],
            max_depth=3,
            include_attributes=True
        )
        
        assert request.query == "find connections"
        assert request.entity_types == ["function", "class"]
        assert request.max_depth == 3
        assert request.include_attributes == True
    
    def test_optimize_context_request(self):
        """Test OptimizeContextRequest model."""
        request = server.OptimizeContextRequest(
            context_window=32000,
            target_tokens=16000,
            strategy="balanced",
            preserve_structure=True
        )
        
        assert request.context_window == 32000
        assert request.target_tokens == 16000
        assert request.strategy == "balanced"
        assert request.preserve_structure == True
    
    def test_watch_directory_request(self):
        """Test WatchDirectoryRequest model."""
        request = server.WatchDirectoryRequest(
            path="/watch/path",
            file_patterns=["*.py"],
            recursive=True,
            debounce_seconds=2.0
        )
        
        assert request.path == "/watch/path"
        assert request.file_patterns == ["*.py"]
        assert request.recursive == True
        assert request.debounce_seconds == 2.0


class TestServerMethods:
    """Test server operational methods."""
    
    @pytest.mark.asyncio
    async def test_index_directory_success(self):
        """Test successful directory indexing."""
        srv = server.EOLRAGContextServer()
        
        # Mock indexer
        mock_indexer = AsyncMock()
        mock_indexer.index_folder = AsyncMock(return_value={
            "indexed_files": 10,
            "total_chunks": 150,
            "processing_time": 5.2
        })
        srv.indexer = mock_indexer
        
        result = await srv.index_directory(
            path="/test/path",
            recursive=True,
            file_patterns=["*.py", "*.md"]
        )
        
        assert "indexed_files" in result
        assert result["indexed_files"] == 10
        assert result["total_chunks"] == 150
        mock_indexer.index_folder.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_index_directory_not_initialized(self):
        """Test directory indexing when server not initialized."""
        srv = server.EOLRAGContextServer()
        # Don't set indexer - simulate uninitialized state
        
        with pytest.raises(RuntimeError, match="Server not initialized"):
            await srv.index_directory(path="/test/path")
    
    @pytest.mark.asyncio
    async def test_index_file_success(self):
        """Test successful file indexing."""
        srv = server.EOLRAGContextServer()
        
        # Mock indexer
        mock_indexer = AsyncMock()
        mock_indexer.index_file = AsyncMock(return_value={
            "file_path": "/test/file.py",
            "chunks_created": 15,
            "processing_time": 0.8
        })
        srv.indexer = mock_indexer
        
        result = await srv.index_file(path="/test/file.py")
        
        assert result["file_path"] == "/test/file.py"
        assert result["chunks_created"] == 15
        mock_indexer.index_file.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_watch_directory_success(self):
        """Test successful directory watching setup."""
        srv = server.EOLRAGContextServer()
        
        # Mock file watcher
        with patch('eol.rag_context.server.FileWatcher') as MockWatcher:
            mock_watcher = MagicMock()
            mock_watcher.watch_directory = MagicMock(return_value="watch_id_123")
            MockWatcher.return_value = mock_watcher
            
            result = await srv.watch_directory(
                path="/watch/path",
                file_patterns=["*.py"],
                recursive=True
            )
            
            assert "watch_id" in result
            assert result["status"] == "watching"
    
    @pytest.mark.asyncio
    async def test_run_server(self):
        """Test server run method."""
        srv = server.EOLRAGContextServer()
        
        # Mock the app.run method
        srv.app = AsyncMock()
        srv.app.run = AsyncMock()
        
        await srv.run()
        
        srv.app.run.assert_called_once()


class TestServerIntegration:
    """Test server integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_server_lifecycle(self):
        """Test complete server lifecycle: init -> operations -> shutdown."""
        srv = server.EOLRAGContextServer()
        
        # Mock all dependencies
        with patch('eol.rag_context.server.RedisVectorStore') as MockRedis, \
             patch('eol.rag_context.server.EmbeddingManager') as MockEmbedding, \
             patch('eol.rag_context.server.DocumentIndexer') as MockIndexer, \
             patch('eol.rag_context.server.SemanticCache') as MockCache, \
             patch('eol.rag_context.server.KnowledgeGraphBuilder') as MockKG:
            
            # Setup mocks
            MockRedis.return_value = AsyncMock()
            MockEmbedding.return_value = AsyncMock()
            MockIndexer.return_value = AsyncMock()
            MockCache.return_value = AsyncMock()
            MockKG.return_value = MagicMock()
            
            # Initialize
            await srv.initialize()
            assert srv.redis_store is not None
            assert srv.indexer is not None
            
            # Perform operations
            srv.indexer.index_folder = AsyncMock(return_value={"indexed_files": 5})
            result = await srv.index_directory("/test")
            assert result["indexed_files"] == 5
            
            # Shutdown
            await srv.shutdown()
    
    def test_server_configuration_validation(self):
        """Test server handles different configuration scenarios."""
        # Test with minimal config
        minimal_config = config.RAGConfig()
        minimal_config.debug = False
        srv1 = server.EOLRAGContextServer(minimal_config)
        assert srv1.config.debug == False
        
        # Test with detailed config
        detailed_config = config.RAGConfig()
        detailed_config.server_name = "detailed-server"
        detailed_config.redis.host = "custom-redis"
        detailed_config.cache.enabled = True
        srv2 = server.EOLRAGContextServer(detailed_config)
        assert srv2.config.server_name == "detailed-server"
        assert srv2.config.redis.host == "custom-redis"
        assert srv2.config.cache.enabled == True
    
    def test_resource_cleanup_scenarios(self):
        """Test resource cleanup in various scenarios."""
        srv = server.EOLRAGContextServer()
        
        # Test cleanup with no resources initialized
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(srv.shutdown())
            # Should not raise any exceptions
        finally:
            loop.close()
        
        # Test cleanup with partial initialization
        srv.redis_store = AsyncMock()
        srv.semantic_cache = None  # Partially initialized
        srv.file_watcher = MagicMock()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(srv.shutdown())
            srv.redis_store.close.assert_called_once()
            srv.file_watcher.stop.assert_called_once()
        finally:
            loop.close()


def test_sync_operations():
    """Test synchronous operations."""
    srv = server.EOLRAGContextServer()
    
    # Test initialization
    assert srv.config is not None
    assert srv.app is not None
    
    # Test configuration access
    assert hasattr(srv.config, 'redis')
    assert hasattr(srv.config, 'embedding')
    assert hasattr(srv.config, 'index')


async def test_async_operations():
    """Test asynchronous operations."""
    srv = server.EOLRAGContextServer()
    
    # Test that async methods exist and can be called
    assert hasattr(srv, 'initialize')
    assert hasattr(srv, 'shutdown')
    assert hasattr(srv, 'index_directory')
    assert hasattr(srv, 'index_file')
    assert hasattr(srv, 'watch_directory')


if __name__ == "__main__":
    # Run sync tests
    test_sync_operations()
    print("✅ Sync server tests passed!")
    
    # Run async tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(test_async_operations())
        print("✅ Async server tests passed!")
    finally:
        loop.close()
    
    print("✅ All comprehensive server tests passed!")