"""
Targeted tests for reaching 80% coverage.
Focus on the most critical uncovered code paths.
"""

import pytest
import sys
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch, PropertyMock, ANY
import numpy as np
import json
import tempfile
import os
from dataclasses import asdict
from collections import deque

# Mock all dependencies
mocks = {}
for module in ['magic', 'pypdf', 'pypdf.PdfReader', 'docx', 'docx.Document',
               'redis', 'redis.asyncio', 'redis.commands', 'redis.commands.search',
               'redis.commands.search.field', 'redis.commands.search.indexDefinition',
               'redis.commands.search.query', 'watchdog', 'watchdog.observers',
               'watchdog.events', 'networkx', 'sentence_transformers', 'openai',
               'tree_sitter', 'tree_sitter_python', 'yaml', 'bs4', 'aiofiles',
               'typer', 'rich', 'rich.console', 'fastmcp', 'fastmcp.server']:
    if module not in sys.modules:
        mocks[module] = MagicMock()
        sys.modules[module] = mocks[module]

# Import modules
from eol.rag_context import config
from eol.rag_context import embeddings
from eol.rag_context import document_processor
from eol.rag_context import indexer
from eol.rag_context import redis_client
from eol.rag_context import semantic_cache
from eol.rag_context import knowledge_graph
from eol.rag_context import file_watcher
from eol.rag_context import server
from eol.rag_context import main


# ============================================================================
# SERVER MODULE - 27% -> 80%+ coverage
# ============================================================================

class TestServerTargeted:
    """Targeted server tests for maximum coverage."""
    
    @pytest.mark.asyncio
    async def test_rag_components_complete(self):
        """Test RAGComponents initialization and methods."""
        with patch('eol.rag_context.server.RedisVectorStore') as MockRedis, \
             patch('eol.rag_context.server.DocumentProcessor') as MockProcessor, \
             patch('eol.rag_context.server.EmbeddingManager') as MockEmbeddings, \
             patch('eol.rag_context.server.DocumentIndexer') as MockIndexer, \
             patch('eol.rag_context.server.SemanticCache') as MockCache, \
             patch('eol.rag_context.server.KnowledgeGraphBuilder') as MockGraph, \
             patch('eol.rag_context.server.FileWatcher') as MockWatcher:
            
            # Create mocks
            mock_redis = MagicMock()
            mock_redis.connect_async = AsyncMock()
            mock_redis.create_indexes = AsyncMock()
            MockRedis.return_value = mock_redis
            
            MockProcessor.return_value = MagicMock()
            MockEmbeddings.return_value = MagicMock()
            MockIndexer.return_value = MagicMock()
            MockCache.return_value = MagicMock()
            MockGraph.return_value = MagicMock()
            MockWatcher.return_value = MagicMock()
            
            # Initialize
            components = server.RAGComponents()
            await components.initialize()
            
            # Verify all components created
            assert components.redis is not None
            assert components.processor is not None
            assert components.embeddings is not None
            assert components.indexer is not None
            assert components.cache is not None
            assert components.graph is not None
            assert components.watcher is not None
            
            mock_redis.connect_async.assert_called_once()
            mock_redis.create_indexes.assert_called_once()
    
    def test_rag_context_server_complete(self):
        """Test RAGContextServer completely."""
        # Mock FastMCP
        mock_mcp = MagicMock()
        mock_mcp_instance = MagicMock()
        mock_mcp.return_value = mock_mcp_instance
        
        # Mock tool decorator
        def mock_tool(name=None):
            def decorator(func):
                func._is_tool = True
                func._tool_name = name or func.__name__
                return func
            return decorator
        
        # Mock resource decorator
        def mock_resource(uri):
            def decorator(func):
                func._is_resource = True
                func._resource_uri = uri
                return func
            return decorator
        
        # Mock prompt decorator
        def mock_prompt(name):
            def decorator(func):
                func._is_prompt = True
                func._prompt_name = name
                return func
            return decorator
        
        mock_mcp_instance.tool = mock_tool
        mock_mcp_instance.resource = mock_resource
        mock_mcp_instance.prompt = mock_prompt
        
        with patch('eol.rag_context.server.FastMCP', mock_mcp), \
             patch('eol.rag_context.server.RAGComponents') as MockComponents:
            
            mock_components = MagicMock()
            MockComponents.return_value = mock_components
            
            # Create server
            srv = server.RAGContextServer()
            assert srv.mcp is not None
            assert srv.components is not None
            
            # Test run method
            mock_mcp_instance.run = MagicMock()
            srv.run()
            mock_mcp_instance.run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_all_server_tools(self):
        """Test all server tools."""
        with patch('eol.rag_context.server.FastMCP'), \
             patch('eol.rag_context.server.RAGComponents') as MockComponents:
            
            mock_components = MagicMock()
            MockComponents.return_value = mock_components
            
            srv = server.RAGContextServer()
            srv.components = mock_components
            
            # Mock components methods
            mock_components.indexer.index_folder = AsyncMock(
                return_value=MagicMock(source_id="src123", file_count=10, total_chunks=50)
            )
            mock_components.redis.search = AsyncMock(
                return_value=[MagicMock(content="Result", metadata={})]
            )
            mock_components.graph.query_subgraph = AsyncMock(
                return_value={"entities": [], "relationships": []}
            )
            mock_components.watcher.watch = AsyncMock(return_value="watch123")
            mock_components.watcher.unwatch = AsyncMock(return_value=True)
            mock_components.cache.get_optimization_report = AsyncMock(
                return_value={"recommendations": [], "current_hit_rate": 0.25}
            )
            mock_components.cache.clear = AsyncMock()
            mock_components.indexer.remove_source = AsyncMock(return_value=True)
            
            # Test all tools
            result = await srv.index_directory("/test")
            assert "src123" in result
            
            result = await srv.search_context("query")
            assert "Result" in result
            
            result = await srv.query_knowledge_graph("entity")
            assert "entities" in result.lower()
            
            result = await srv.watch_directory("/test")
            assert "watch123" in result
            
            result = await srv.unwatch_directory("watch123")
            assert "unwatched" in result.lower()
            
            result = await srv.optimize_context()
            assert "0.25" in result or "25" in result
            
            result = await srv.clear_cache()
            assert "cleared" in result.lower()
            
            result = await srv.remove_source("src123")
            assert "removed" in result.lower()
    
    @pytest.mark.asyncio
    async def test_all_server_resources(self):
        """Test all server resources."""
        with patch('eol.rag_context.server.FastMCP'), \
             patch('eol.rag_context.server.RAGComponents') as MockComponents:
            
            mock_components = MagicMock()
            MockComponents.return_value = mock_components
            
            srv = server.RAGContextServer()
            srv.components = mock_components
            
            # Mock methods
            mock_components.redis.get_context = AsyncMock(
                return_value=[MagicMock(content="Context", metadata={})]
            )
            mock_components.indexer.list_sources = AsyncMock(
                return_value=[{"source_id": "src1", "path": "/path1"}]
            )
            mock_components.indexer.get_stats = MagicMock(return_value={"docs": 100})
            mock_components.cache.get_stats = MagicMock(return_value={"hit_rate": 0.31})
            mock_components.graph.get_graph_stats = MagicMock(return_value={"entities": 50})
            
            # Test resources
            result = await srv.get_context("context://query")
            assert "Context" in result
            
            result = await srv.list_sources()
            assert "src1" in result
            
            result = await srv.get_stats()
            assert "100" in result or "0.31" in result or "50" in result
    
    @pytest.mark.asyncio
    async def test_server_prompts(self):
        """Test server prompts."""
        with patch('eol.rag_context.server.FastMCP'), \
             patch('eol.rag_context.server.RAGComponents') as MockComponents:
            
            mock_components = MagicMock()
            MockComponents.return_value = mock_components
            
            srv = server.RAGContextServer()
            srv.components = mock_components
            
            mock_components.redis.search = AsyncMock(return_value=[])
            
            result = await srv.structured_query("Find information")
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_server_error_handling(self):
        """Test server error handling."""
        with patch('eol.rag_context.server.FastMCP'), \
             patch('eol.rag_context.server.RAGComponents') as MockComponents:
            
            mock_components = MagicMock()
            MockComponents.return_value = mock_components
            
            srv = server.RAGContextServer()
            srv.components = mock_components
            
            # Test errors in tools
            mock_components.indexer.index_folder = AsyncMock(
                side_effect=Exception("Index error")
            )
            
            result = await srv.index_directory("/test")
            assert "error" in result.lower()
            assert "Index error" in result


# ============================================================================
# MAIN MODULE - 20% -> 80%+ coverage
# ============================================================================

class TestMainTargeted:
    """Targeted main module tests."""
    
    def test_main_cli_complete(self):
        """Test main CLI completely."""
        # Test app creation
        assert main.app is not None
        
        # Mock console
        with patch('eol.rag_context.main.console') as mock_console:
            mock_console.print = MagicMock()
            
            # Test serve command
            with patch('eol.rag_context.main.RAGContextServer') as MockServer:
                mock_server = MagicMock()
                MockServer.return_value = mock_server
                
                main.serve(
                    host="0.0.0.0",
                    port=9000,
                    data_dir="/custom/data",
                    index_dir="/custom/index"
                )
                
                MockServer.assert_called_once()
                mock_server.run.assert_called_once()
            
            # Test index command
            with patch('eol.rag_context.main.DocumentIndexer') as MockIndexer, \
                 patch('eol.rag_context.main.RAGComponents') as MockComponents, \
                 patch('eol.rag_context.main.asyncio') as mock_asyncio:
                
                mock_components = MagicMock()
                MockComponents.return_value = mock_components
                mock_indexer = MagicMock()
                MockIndexer.return_value = mock_indexer
                mock_asyncio.run = MagicMock()
                
                main.index(
                    path="/test/path",
                    recursive=True,
                    patterns=["*.py", "*.md", "*.txt"]
                )
                
                mock_asyncio.run.assert_called_once()
                mock_console.print.assert_called()
            
            # Test search command
            with patch('eol.rag_context.main.RedisVectorStore') as MockRedis, \
                 patch('eol.rag_context.main.RAGComponents') as MockComponents, \
                 patch('eol.rag_context.main.asyncio') as mock_asyncio:
                
                mock_components = MagicMock()
                MockComponents.return_value = mock_components
                mock_redis = MagicMock()
                MockRedis.return_value = mock_redis
                mock_asyncio.run = MagicMock()
                
                main.search(
                    query="test query",
                    limit=10,
                    hierarchy_level=2
                )
                
                mock_asyncio.run.assert_called_once()
            
            # Test stats command
            with patch('eol.rag_context.main.RAGComponents') as MockComponents, \
                 patch('eol.rag_context.main.asyncio') as mock_asyncio:
                
                mock_components = MagicMock()
                MockComponents.return_value = mock_components
                mock_asyncio.run = MagicMock()
                
                main.stats()
                
                mock_asyncio.run.assert_called_once()
            
            # Test clear-cache command
            with patch('eol.rag_context.main.SemanticCache') as MockCache, \
                 patch('eol.rag_context.main.RAGComponents') as MockComponents, \
                 patch('eol.rag_context.main.asyncio') as mock_asyncio:
                
                mock_components = MagicMock()
                MockComponents.return_value = mock_components
                mock_cache = MagicMock()
                MockCache.return_value = mock_cache
                mock_asyncio.run = MagicMock()
                
                main.clear_cache()
                
                mock_asyncio.run.assert_called_once()
            
            # Test watch command
            with patch('eol.rag_context.main.FileWatcher') as MockWatcher, \
                 patch('eol.rag_context.main.RAGComponents') as MockComponents, \
                 patch('eol.rag_context.main.asyncio') as mock_asyncio:
                
                mock_components = MagicMock()
                MockComponents.return_value = mock_components
                mock_watcher = MagicMock()
                MockWatcher.return_value = mock_watcher
                mock_asyncio.run = MagicMock()
                
                main.watch(
                    path="/test/path",
                    recursive=True,
                    patterns=["*.py"]
                )
                
                mock_asyncio.run.assert_called_once()
            
            # Test main entry point
            with patch.object(main.app, 'run') as mock_run:
                main.main()
                mock_run.assert_called_once()


# ============================================================================
# SEMANTIC CACHE - 47% -> 80%+ coverage
# ============================================================================

class TestSemanticCacheTargeted:
    """Targeted semantic cache tests."""
    
    @pytest.mark.asyncio
    async def test_cache_complete_lifecycle(self):
        """Test complete cache lifecycle."""
        cfg = MagicMock()
        cfg.enabled = True
        cfg.similarity_threshold = 0.9
        cfg.adaptive_threshold = True
        cfg.max_cache_size = 3
        cfg.ttl_seconds = 3600
        cfg.target_hit_rate = 0.31
        
        emb = MagicMock()
        emb.get_embedding = AsyncMock(return_value=np.random.randn(128))
        
        redis = MagicMock()
        redis.redis = MagicMock()
        redis.redis.hset = AsyncMock()
        redis.redis.expire = AsyncMock()
        redis.redis.hlen = AsyncMock(return_value=2)
        redis.redis.hgetall = AsyncMock(return_value={
            "cache:1": json.dumps({"query": "q1", "response": "r1", "timestamp": 1}),
            "cache:2": json.dumps({"query": "q2", "response": "r2", "timestamp": 2})
        })
        redis.redis.hdel = AsyncMock()
        redis.redis.delete = AsyncMock()
        redis.redis.keys = AsyncMock(return_value=["cache:1", "cache:2"])
        redis.redis.hincrby = AsyncMock()
        redis.redis.ft = MagicMock()
        redis.redis.ft.return_value.search = AsyncMock(return_value=MagicMock(docs=[]))
        
        cache = semantic_cache.SemanticCache(cfg, emb, redis)
        
        # Test set
        await cache.set("query1", "response1", {"meta": "data"})
        redis.redis.hset.assert_called()
        
        # Test get miss
        result = await cache.get("query1")
        assert cache.stats["misses"] > 0
        
        # Test get hit
        mock_doc = MagicMock()
        mock_doc.id = "cache:123"
        mock_doc.score = 0.95
        mock_doc.response = "cached"
        mock_doc.metadata = json.dumps({"key": "val"})
        mock_doc.hit_count = 5
        
        redis.redis.ft.return_value.search = AsyncMock(
            return_value=MagicMock(docs=[mock_doc])
        )
        
        result = await cache.get("similar")
        assert result is not None
        assert cache.stats["hits"] > 0
        
        # Test eviction
        redis.redis.hlen = AsyncMock(return_value=4)  # Over limit
        await cache._evict_oldest()
        redis.redis.hdel.assert_called()
        
        # Test clear
        await cache.clear()
        redis.redis.delete.assert_called()
        
        # Test optimization
        cache.similarity_scores = [0.8, 0.85, 0.9, 0.95] * 10
        cache.stats = {"queries": 100, "hits": 25}
        
        await cache._update_adaptive_threshold()
        assert cache.adaptive_threshold != 0.9
        
        async def mock_size():
            return 10
        cache._get_cache_size = mock_size
        
        report = await cache.get_optimization_report()
        assert "recommendations" in report
        assert "current_hit_rate" in report
        assert "percentiles" in report
    
    @pytest.mark.asyncio
    async def test_cache_disabled(self):
        """Test cache when disabled."""
        cfg = MagicMock()
        cfg.enabled = False
        
        cache = semantic_cache.SemanticCache(cfg, MagicMock(), MagicMock())
        
        result = await cache.get("query")
        assert result is None
        
        await cache.set("query", "response", {})
        # Should not crash
        
        await cache.clear()
        # Should not crash


# ============================================================================
# FILE WATCHER - 38% -> 80%+ coverage
# ============================================================================

class TestFileWatcherTargeted:
    """Targeted file watcher tests."""
    
    @pytest.mark.asyncio
    async def test_watcher_complete_lifecycle(self):
        """Test complete watcher lifecycle."""
        idx = MagicMock()
        idx.scanner = MagicMock()
        idx.scanner.generate_source_id = MagicMock(return_value="src123")
        idx.index_folder = AsyncMock(return_value=MagicMock(
            source_id="src123", file_count=5, total_chunks=20
        ))
        idx.index_file = AsyncMock()
        idx.remove_source = AsyncMock()
        
        watcher = file_watcher.FileWatcher(idx, debounce_seconds=0, batch_size=2)
        
        with patch('eol.rag_context.file_watcher.Observer') as MockObserver:
            mock_observer = MagicMock()
            mock_observer.is_alive = MagicMock(return_value=True)
            mock_observer.stop = MagicMock()
            mock_observer.join = MagicMock()
            MockObserver.return_value = mock_observer
            
            # Start
            await watcher.start()
            assert watcher.is_running
            mock_observer.start.assert_called()
            
            # Watch directory
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir = Path(tmpdir)
                
                source_id = await watcher.watch(
                    tmpdir,
                    recursive=True,
                    file_patterns=["*.py", "*.md"],
                    ignore_patterns=["*.pyc"]
                )
                assert source_id == "src123"
                assert source_id in watcher.watched_sources
                
                # Get watched source
                source = watcher.watched_sources[source_id]
                assert source.recursive is True
                assert "*.py" in source.file_patterns
                
                # Simulate events
                handler = watcher.observers.get(source_id)
                assert handler is not None
                
                # Test all event types
                event = MagicMock()
                event.is_directory = False
                event.src_path = str(tmpdir / "test.py")
                
                # Created
                event.event_type = "created"
                handler.on_created(event)
                
                # Modified
                event.event_type = "modified"
                handler.on_modified(event)
                
                # Deleted
                event.event_type = "deleted"
                handler.on_deleted(event)
                
                # Moved
                event.event_type = "moved"
                event.dest_path = str(tmpdir / "new.py")
                handler.on_moved(event)
                
                # Process queue
                assert len(watcher.change_queue) > 0
                await watcher._process_changes()
                assert watcher.stats["changes_processed"] > 0
                
                # Test callbacks
                called = []
                def callback(change):
                    called.append(change)
                
                watcher.add_change_callback(callback)
                
                # Add change and process
                watcher.change_queue.append(
                    file_watcher.FileChange(
                        path=Path("/test.py"),
                        change_type=file_watcher.ChangeType.MODIFIED
                    )
                )
                await watcher._process_changes()
                
                # Remove callback
                watcher.remove_change_callback(callback)
                
                # Get history
                history = watcher.get_change_history(limit=5)
                assert len(history) > 0
                
                # Unwatch
                success = await watcher.unwatch(source_id)
                assert success
                assert source_id not in watcher.watched_sources
                
                # Unwatch non-existent
                success = await watcher.unwatch("nonexistent")
                assert not success
            
            # Stop
            await watcher.stop()
            assert not watcher.is_running
            mock_observer.stop.assert_called()
            
            # Test stop when not running
            watcher.is_running = False
            await watcher.stop()  # Should not crash
    
    @pytest.mark.asyncio
    async def test_watcher_error_handling(self):
        """Test watcher error handling."""
        idx = MagicMock()
        idx.index_file = AsyncMock(side_effect=Exception("Index error"))
        
        watcher = file_watcher.FileWatcher(idx)
        
        # Process with error
        watcher.change_queue.append(
            file_watcher.FileChange(
                path=Path("/test.py"),
                change_type=file_watcher.ChangeType.MODIFIED
            )
        )
        
        await watcher._process_changes()
        assert watcher.stats["errors"] > 0
    
    def test_change_event_handler(self):
        """Test ChangeEventHandler completely."""
        callback = MagicMock()
        handler = file_watcher.ChangeEventHandler(
            Path("/test"),
            callback,
            file_patterns=["*.py", "*.md"],
            ignore_patterns=["*.pyc", "__pycache__/*"]
        )
        
        # Test matching
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/file.py"
        event.event_type = "created"
        
        handler.on_created(event)
        callback.assert_called_once()
        
        # Test ignored
        event.src_path = "/test/file.pyc"
        handler.on_created(event)
        callback.assert_called_once()  # Not called again
        
        # Test directory
        event.is_directory = True
        event.src_path = "/test/dir"
        handler.on_created(event)
        callback.assert_called_once()  # Not called for directory
        
        # Test non-matching pattern
        event.is_directory = False
        event.src_path = "/test/file.txt"
        handler.on_created(event)
        callback.assert_called_once()  # Not called for non-matching


# ============================================================================
# KNOWLEDGE GRAPH - 42% -> 80%+ coverage
# ============================================================================

class TestKnowledgeGraphTargeted:
    """Targeted knowledge graph tests."""
    
    @pytest.mark.asyncio
    async def test_graph_complete_pipeline(self):
        """Test complete graph building pipeline."""
        redis = MagicMock()
        redis.store_entities = AsyncMock()
        redis.store_relationships = AsyncMock()
        
        emb = MagicMock()
        emb.get_embeddings = AsyncMock(
            side_effect=lambda texts, **kwargs: np.random.randn(len(texts), 128)
        )
        
        builder = knowledge_graph.KnowledgeGraphBuilder(redis, emb)
        
        # Build from various documents
        docs = [
            redis_client.VectorDocument(
                id="doc1",
                content="""# Main Architecture
                
## Components
- UserService handles authentication
- DataProcessor manages data pipeline
- APIGateway routes requests

## Dependencies
Requires Redis, PostgreSQL, and RabbitMQ

## APIs
- `/api/users` - User management
- `/api/data` - Data operations

Links to [documentation](http://docs.example.com)
""",
                embedding=np.random.randn(128),
                metadata={"doc_type": "markdown", "relative_path": "architecture.md"},
                hierarchy_level=3
            ),
            redis_client.VectorDocument(
                id="doc2",
                content="""
import redis
import psycopg2
from typing import List, Dict

class UserService:
    '''Handles user authentication and management'''
    
    def __init__(self):
        self.db = psycopg2.connect("postgresql://localhost/users")
        self.cache = redis.Redis()
    
    def authenticate(self, username: str, password: str) -> bool:
        '''Authenticate a user'''
        return self._check_credentials(username, password)
    
    def create_user(self, user_data: Dict) -> str:
        '''Create a new user'''
        return self._store_user(user_data)

class DataProcessor:
    '''Manages data processing pipeline'''
    
    def process_batch(self, data: List) -> List:
        '''Process a batch of data'''
        return [self._process_item(item) for item in data]

def main():
    service = UserService()
    processor = DataProcessor()
    # Application logic
""",
                embedding=np.random.randn(128),
                metadata={"doc_type": "code", "language": "python", "relative_path": "services.py"},
                hierarchy_level=3
            ),
            redis_client.VectorDocument(
                id="doc3",
                content=json.dumps({
                    "services": {
                        "user_service": {
                            "endpoint": "/api/users",
                            "methods": ["GET", "POST", "PUT", "DELETE"],
                            "dependencies": ["postgresql", "redis"]
                        },
                        "data_service": {
                            "endpoint": "/api/data",
                            "methods": ["GET", "POST"],
                            "dependencies": ["rabbitmq", "redis"]
                        }
                    },
                    "features": ["authentication", "data-processing", "caching"],
                    "version": "1.0.0"
                }),
                embedding=np.random.randn(128),
                metadata={"doc_type": "json", "relative_path": "config.json"},
                hierarchy_level=3
            )
        ]
        
        await builder.build_from_documents(docs)
        
        # Should have extracted various entities
        assert len(builder.entities) > 0
        assert len(builder.relationships) > 0
        
        # Check entity types
        entity_types = {e.type for e in builder.entities.values()}
        assert any(t in entity_types for t in [
            knowledge_graph.EntityType.CLASS,
            knowledge_graph.EntityType.FUNCTION,
            knowledge_graph.EntityType.API,
            knowledge_graph.EntityType.DEPENDENCY,
            knowledge_graph.EntityType.FEATURE
        ])
        
        # Discover patterns
        await builder._discover_patterns()
        
        # Query subgraph
        subgraph = await builder.query_subgraph("UserService", max_depth=2)
        assert "entities" in subgraph
        assert "relationships" in subgraph
        
        # Export graph
        graph_data = await builder.export_graph()
        assert "nodes" in graph_data
        assert "edges" in graph_data
        assert len(graph_data["nodes"]) > 0
        assert len(graph_data["edges"]) > 0
        
        # Test stats
        stats = builder.get_graph_stats()
        assert stats["entity_count"] > 0
        assert stats["relationship_count"] > 0
        assert len(stats["entity_types"]) > 0
        assert len(stats["relationship_types"]) > 0


# ============================================================================
# REDIS CLIENT - 28% -> 80%+ coverage
# ============================================================================

class TestRedisClientTargeted:
    """Targeted Redis client tests."""
    
    @pytest.mark.asyncio
    async def test_redis_complete_operations(self):
        """Test all Redis operations."""
        store = redis_client.RedisVectorStore(
            config.RedisConfig(host="localhost", port=6379, password="test"),
            config.IndexConfig(name="test_index", distance_metric="L2")
        )
        
        with patch('eol.rag_context.redis_client.AsyncRedis') as MockAsyncRedis, \
             patch('eol.rag_context.redis_client.Redis') as MockSyncRedis:
            
            # Mock async Redis
            mock_async = MagicMock()
            async def mock_connect(*args, **kwargs):
                return mock_async
            MockAsyncRedis.side_effect = mock_connect
            
            mock_async.ping = AsyncMock(return_value=True)
            mock_async.ft = MagicMock()
            mock_async.ft.return_value.info = AsyncMock(side_effect=Exception("No index"))
            mock_async.ft.return_value.create_index = AsyncMock()
            mock_async.hset = AsyncMock()
            mock_async.keys = AsyncMock(return_value=["doc:1", "doc:2"])
            mock_async.delete = AsyncMock()
            
            # Mock sync Redis
            mock_sync = MagicMock()
            mock_sync.ping = MagicMock(return_value=True)
            MockSyncRedis.return_value = mock_sync
            
            # Connect
            await store.connect_async()
            assert store.async_redis is not None
            
            store.connect_sync()
            assert store.redis is not None
            
            # Create indexes
            await store.create_indexes()
            mock_async.ft.return_value.create_index.assert_called()
            
            # Store document
            doc = redis_client.VectorDocument(
                id="test123",
                content="Test content",
                embedding=np.array([1.0, 2.0, 3.0]),
                metadata={"key": "value"},
                hierarchy_level=2,
                parent_id="parent",
                children_ids=["child1", "child2"]
            )
            
            await store.store_document(doc)
            mock_async.hset.assert_called()
            
            # Mock search
            mock_doc = MagicMock()
            mock_doc.id = "doc:1"
            mock_doc.content = "Found content"
            mock_doc.metadata = json.dumps({"score": 0.9})
            mock_doc.embedding = np.array([1, 2, 3]).tobytes()
            mock_doc.hierarchy_level = 2
            
            mock_result = MagicMock()
            mock_result.docs = [mock_doc]
            mock_result.total = 1
            
            mock_async.ft.return_value.search = AsyncMock(return_value=mock_result)
            
            # Search
            results = await store.search("query", limit=5, hierarchy_level=2)
            assert len(results) == 1
            assert results[0].content == "Found content"
            
            # Get context (hierarchical)
            mock_async.ft.return_value.search = AsyncMock(side_effect=[
                mock_result,  # Level 1
                mock_result,  # Level 2
                mock_result   # Level 3
            ])
            
            context = await store.get_context("query", max_chunks=10)
            assert len(context) > 0
            
            # Delete by source
            await store.delete_by_source("src123")
            mock_async.delete.assert_called()
            
            # Store entities
            entities = [
                knowledge_graph.Entity("e1", "Entity1", knowledge_graph.EntityType.CLASS),
                knowledge_graph.Entity("e2", "Entity2", knowledge_graph.EntityType.FUNCTION)
            ]
            await store.store_entities(entities)
            assert mock_async.hset.call_count > 1
            
            # Store relationships
            relationships = [
                knowledge_graph.Relationship("e1", "e2", knowledge_graph.RelationType.CALLS)
            ]
            await store.store_relationships(relationships)
            assert mock_async.hset.call_count > 1
            
            # List sources
            mock_async.keys = AsyncMock(return_value=["source:src1", "source:src2"])
            mock_async.hgetall = AsyncMock(return_value={
                b"path": b"/path1",
                b"indexed_at": b"123.0"
            })
            
            sources = await store.list_sources()
            assert isinstance(sources, list)


# ============================================================================
# Run all targeted tests
# ============================================================================

@pytest.mark.asyncio
async def test_run_all_targeted():
    """Run all targeted tests."""
    # This ensures all test classes are executed
    test_instances = [
        TestServerTargeted(),
        TestMainTargeted(),
        TestSemanticCacheTargeted(),
        TestFileWatcherTargeted(),
        TestKnowledgeGraphTargeted(),
        TestRedisClientTargeted(),
    ]
    
    for instance in test_instances:
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                method = getattr(instance, method_name)
                if asyncio.iscoroutinefunction(method):
                    await method()
                elif callable(method):
                    method()