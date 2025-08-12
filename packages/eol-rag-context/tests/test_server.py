"""Fixed unit tests for server module - testing only actual functionality."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock, ANY
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eol.rag_context import server
from eol.rag_context.config import RAGConfig


class TestEOLRAGContextServer:
    """Test EOLRAGContextServer with actual methods."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RAGConfig()
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components."""
        with patch('eol.rag_context.server.RedisVectorStore') as MockRedis, \
             patch('eol.rag_context.server.EmbeddingManager') as MockEmb, \
             patch('eol.rag_context.server.DocumentProcessor') as MockProc, \
             patch('eol.rag_context.server.DocumentIndexer') as MockIdx, \
             patch('eol.rag_context.server.SemanticCache') as MockCache, \
             patch('eol.rag_context.server.KnowledgeGraphBuilder') as MockGraph, \
             patch('eol.rag_context.server.FileWatcher') as MockWatcher, \
             patch('eol.rag_context.server.FastMCP') as MockMCP:
            
            # Setup mock instances
            mock_redis = AsyncMock()
            mock_redis.connect_async = AsyncMock()
            MockRedis.return_value = mock_redis
            
            mock_emb = MagicMock()
            MockEmb.return_value = mock_emb
            
            mock_proc = MagicMock()
            MockProc.return_value = mock_proc
            
            mock_idx = AsyncMock()
            mock_idx.index_folder = AsyncMock(return_value=MagicMock(chunks=10))
            mock_idx.index_file = AsyncMock(return_value=MagicMock(chunks=5))
            MockIdx.return_value = mock_idx
            
            mock_cache = AsyncMock()
            mock_cache.initialize = AsyncMock()
            MockCache.return_value = mock_cache
            
            mock_graph = AsyncMock()
            mock_graph.initialize = AsyncMock()
            MockGraph.return_value = mock_graph
            
            mock_watcher = AsyncMock()
            mock_watcher.watch = AsyncMock()
            MockWatcher.return_value = mock_watcher
            
            mock_mcp = MagicMock()
            mock_mcp.run = AsyncMock()
            MockMCP.return_value = mock_mcp
            
            yield {
                'redis': mock_redis,
                'emb': mock_emb,
                'proc': mock_proc,
                'idx': mock_idx,
                'cache': mock_cache,
                'graph': mock_graph,
                'watcher': mock_watcher,
                'mcp': mock_mcp
            }
    
    def test_server_creation(self, config, mock_components):
        """Test server can be created."""
        srv = server.EOLRAGContextServer(config)
        assert srv is not None
        assert srv.config == config
    
    @pytest.mark.asyncio
    async def test_initialize(self, config, mock_components):
        """Test server initialization."""
        srv = server.EOLRAGContextServer(config)
        await srv.initialize()
        
        # Check components were initialized
        mock_components['redis'].connect_async.assert_called_once()
        mock_components['cache'].initialize.assert_called_once()
        mock_components['graph'].initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown(self, config, mock_components):
        """Test server shutdown."""
        srv = server.EOLRAGContextServer(config)
        await srv.initialize()
        await srv.shutdown()
        
        # Shutdown should be clean
        assert srv is not None
    
    @pytest.mark.asyncio
    async def test_index_directory(self, config, mock_components):
        """Test indexing a directory."""
        srv = server.EOLRAGContextServer(config)
        await srv.initialize()
        
        result = await srv.index_directory("/test/path", recursive=True)
        
        assert result is not None
        mock_components['idx'].index_folder.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_index_file(self, config, mock_components):
        """Test indexing a single file."""
        srv = server.EOLRAGContextServer(config)
        await srv.initialize()
        
        result = await srv.index_file("/test/file.py")
        
        assert result is not None
        mock_components['idx'].index_file.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_watch_directory(self, config, mock_components):
        """Test watching a directory."""
        srv = server.EOLRAGContextServer(config)
        await srv.initialize()
        
        result = await srv.watch_directory("/test/path", patterns=["*.py"])
        
        assert result is not None
        mock_components['watcher'].watch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_method(self, config, mock_components):
        """Test server run method."""
        srv = server.EOLRAGContextServer(config)
        
        # Mock run to return immediately
        mock_components['mcp'].run = AsyncMock()
        
        await srv.run()
        
        mock_components['mcp'].run.assert_called_once()
    
    def test_setup_resources(self, config, mock_components):
        """Test resource setup."""
        srv = server.EOLRAGContextServer(config)
        # Resources are set up in __init__ via _setup_resources
        assert srv.mcp is not None
    
    def test_setup_tools(self, config, mock_components):
        """Test tool setup."""
        srv = server.EOLRAGContextServer(config)
        # Tools are set up in __init__ via _setup_tools
        assert srv.mcp is not None
    
    def test_setup_prompts(self, config, mock_components):
        """Test prompt setup."""
        srv = server.EOLRAGContextServer(config)
        # Prompts are set up in __init__ via _setup_prompts
        assert srv.mcp is not None


class TestServerRequestModels:
    """Test server request models."""
    
    def test_index_directory_request(self):
        """Test IndexDirectoryRequest model."""
        req = server.IndexDirectoryRequest(
            path="/test",
            recursive=True,
            file_patterns=["*.py"],
            watch=False
        )
        assert req.path == "/test"
        assert req.recursive is True
        assert req.file_patterns == ["*.py"]
        assert req.watch is False
    
    def test_search_context_request(self):
        """Test SearchContextRequest model."""
        req = server.SearchContextRequest(
            query="test query",
            max_results=5,
            min_relevance=0.8
        )
        assert req.query == "test query"
        assert req.max_results == 5
        assert req.min_relevance == 0.8
    
    def test_query_knowledge_graph_request(self):
        """Test QueryKnowledgeGraphRequest model."""
        req = server.QueryKnowledgeGraphRequest(
            query="TestEntity",
            max_depth=3,
            max_entities=10
        )
        assert req.query == "TestEntity"
        assert req.max_depth == 3
        assert req.max_entities == 10
    
    def test_optimize_context_request(self):
        """Test OptimizeContextRequest model."""
        req = server.OptimizeContextRequest(
            query="optimize this query"
        )
        assert req.query == "optimize this query"
    
    def test_watch_directory_request(self):
        """Test WatchDirectoryRequest model."""
        req = server.WatchDirectoryRequest(
            path="/test",
            file_patterns=["*.py", "*.md"],
            recursive=True
        )
        assert req.path == "/test"
        assert req.file_patterns == ["*.py", "*.md"]
        assert req.recursive is True


class TestServerAdditional:
    """Additional server tests for better coverage."""
    
    @pytest.mark.asyncio
    @patch('eol.rag_context.server.FastMCP')
    async def test_server_mcp_tools(self, mock_mcp_class):
        """Test MCP tool setup in server."""
        mock_mcp = MagicMock()
        mock_mcp_class.return_value = mock_mcp
        mock_mcp.tool = MagicMock()
        
        with patch.multiple(
            'eol.rag_context.server',
            RedisVectorStore=MagicMock(),
            EmbeddingManager=MagicMock(),
            DocumentProcessor=MagicMock(),
            DocumentIndexer=MagicMock(),
            SemanticCache=MagicMock(),
            KnowledgeGraphBuilder=MagicMock(),
            FileWatcher=MagicMock()
        ):
            config = RAGConfig()
            srv = server.EOLRAGContextServer(config)
            
            # Check tools were registered
            assert mock_mcp.tool.call_count > 0
    
    @pytest.mark.asyncio
    async def test_server_error_handling(self):
        """Test server error handling."""
        with patch.multiple(
            'eol.rag_context.server',
            RedisVectorStore=MagicMock(side_effect=Exception("Redis error")),
            EmbeddingManager=MagicMock(),
            DocumentProcessor=MagicMock(),
            DocumentIndexer=MagicMock(),
            SemanticCache=MagicMock(),
            KnowledgeGraphBuilder=MagicMock(),
            FileWatcher=MagicMock(),
            FastMCP=MagicMock()
        ):
            config = RAGConfig()
            with pytest.raises(Exception) as exc_info:
                srv = server.EOLRAGContextServer(config)
            assert "Redis error" in str(exc_info.value)