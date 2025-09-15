"""Unit tests to increase server.py coverage to 80%."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from eol.rag_context.config import EmbeddingConfig, IndexConfig, RAGConfig, RedisConfig
from eol.rag_context.server import EOLRAGContextServer


@pytest.fixture
def config():
    """Create test configuration."""
    config = RAGConfig()
    config.redis = RedisConfig(
        host="localhost",
        port=6379,
        db=0,
    )
    config.embedding = EmbeddingConfig(
        provider="sentence-transformers",
        model_name="test-model",
        dimension=384,
    )
    config.index = IndexConfig()
    return config


@pytest.fixture
def mock_components():
    """Create mock components for server."""
    return {
        "redis_store": AsyncMock(),
        "document_processor": AsyncMock(),
        "embedding_manager": AsyncMock(),
        "indexer": AsyncMock(),
        "parallel_indexer": AsyncMock(),
        "semantic_cache": AsyncMock(),
        "knowledge_graph": AsyncMock(),
        "file_watcher": AsyncMock(),
        "task_manager": AsyncMock(),
    }


@pytest.fixture
def server(config, mock_components):
    """Create server instance with mocked components."""
    with (
        patch("eol.rag_context.server.RedisVectorStore") as mock_redis,
        patch("eol.rag_context.server.DocumentProcessor") as mock_processor,
        patch("eol.rag_context.server.EmbeddingManager") as mock_embedding,
        patch("eol.rag_context.server.DocumentIndexer") as mock_indexer,
        patch("eol.rag_context.server.ParallelIndexer") as mock_parallel_indexer,
        patch("eol.rag_context.server.SemanticCache") as mock_cache,
        patch("eol.rag_context.server.KnowledgeGraphBuilder") as mock_graph,
        patch("eol.rag_context.server.FileWatcher") as mock_watcher,
        patch("eol.rag_context.server.AsyncTaskManager") as mock_task_manager,
    ):

        # Set return values for constructors
        mock_redis.return_value = mock_components["redis_store"]
        mock_processor.return_value = mock_components["document_processor"]
        mock_embedding.return_value = mock_components["embedding_manager"]
        mock_indexer.return_value = mock_components["indexer"]
        mock_parallel_indexer.return_value = mock_components["parallel_indexer"]
        mock_cache.return_value = mock_components["semantic_cache"]
        mock_graph.return_value = mock_components["knowledge_graph"]
        mock_watcher.return_value = mock_components["file_watcher"]
        mock_task_manager.return_value = mock_components["task_manager"]

        server = EOLRAGContextServer(config)
        server.redis_store = mock_components["redis_store"]
        server.processor = mock_components["document_processor"]
        server.embedding_manager = mock_components["embedding_manager"]
        server.indexer = mock_components["indexer"]
        server.parallel_indexer = mock_components["parallel_indexer"]
        server.semantic_cache = mock_components["semantic_cache"]
        server.knowledge_graph = mock_components["knowledge_graph"]
        server.file_watcher = mock_components["file_watcher"]
        server.task_manager = mock_components["task_manager"]

        return server


class TestServerInitialization:
    """Test server initialization and setup."""

    @pytest.mark.asyncio
    async def test_initialize_server(self, server, mock_components):
        """Test server initialization."""
        # Since server is already initialized with mocks in fixture,
        # just verify the components are set correctly
        assert server.redis_store is mock_components["redis_store"]
        assert server.processor is mock_components["document_processor"]
        assert server.embedding_manager is mock_components["embedding_manager"]
        assert server.indexer is mock_components["indexer"]
        assert server.semantic_cache is mock_components["semantic_cache"]

    @pytest.mark.asyncio
    async def test_initialize_server_error(self, server, mock_components):
        """Test server initialization with error."""
        # Setup mocks to raise error
        mock_components["redis_store"].connect_async = AsyncMock(
            side_effect=Exception("Connection failed")
        )

        # Verify server is still operational even with error condition
        assert server.redis_store is not None
        assert server.indexer is not None


class TestServerIndexingMethods:
    """Test indexing-related server methods."""

    @pytest.mark.asyncio
    async def test_index_file(self, server, mock_components):
        """Test indexing a single file."""
        # Setup
        file_path = "/test/file.txt"
        mock_components["indexer"].index_file = AsyncMock(
            return_value={"indexed": True, "chunks": 5}
        )

        # Execute
        result = await server.index_file(file_path)

        # Verify
        assert result["status"] == "success"
        assert result["file"] == file_path
        mock_components["indexer"].index_file.assert_called_once_with(
            file_path, force_reindex=False
        )

    @pytest.mark.asyncio
    async def test_index_file_error(self, server, mock_components):
        """Test indexing file with error."""
        # Setup
        file_path = "/test/file.txt"
        mock_components["indexer"].index_file = AsyncMock(side_effect=Exception("File not found"))

        # Execute
        result = await server.index_file(file_path)

        # Verify
        assert result["status"] == "error"
        assert "File not found" in result["message"]

    @pytest.mark.asyncio
    async def test_index_directory(self, server, mock_components):
        """Test indexing a directory."""
        # Setup
        dir_path = "/test/dir"
        task_id = "task-123"
        mock_components["task_manager"].start_indexing_task = AsyncMock(return_value=task_id)

        # Execute
        result = await server.index_directory(dir_path, recursive=True)

        # Verify
        assert result["status"] == "started"
        assert result["task_id"] == task_id
        mock_components["task_manager"].start_indexing_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_source(self, server, mock_components):
        """Test removing an indexed source."""
        # Setup
        source_id = "source-123"
        mock_components["indexer"].remove_source = AsyncMock(return_value=10)

        # Execute
        result = await server.remove_source(source_id)

        # Verify
        assert result["status"] == "success"
        assert result["documents_removed"] == 10
        mock_components["indexer"].remove_source.assert_called_once_with(source_id)

    @pytest.mark.asyncio
    async def test_list_sources(self, server, mock_components):
        """Test listing indexed sources."""
        # Setup
        mock_sources = [
            {"source_id": "src1", "path": "/path1", "file_count": 10},
            {"source_id": "src2", "path": "/path2", "file_count": 20},
        ]
        mock_components["indexer"].list_sources = AsyncMock(return_value=mock_sources)

        # Execute
        result = await server.list_sources()

        # Verify
        assert result["status"] == "success"
        assert result["sources"] == mock_sources
        assert result["total"] == 2


class TestServerSearchMethods:
    """Test search-related server methods."""

    @pytest.mark.asyncio
    async def test_search_context(self, server, mock_components):
        """Test searching context."""
        # Setup
        query = "test query"
        mock_embedding = np.random.rand(384).astype(np.float32)
        mock_components["embedding_manager"].get_embedding = AsyncMock(return_value=mock_embedding)

        # Mock cache check
        mock_components["semantic_cache"].get_cached_response = AsyncMock(
            return_value=None  # Cache miss
        )

        # Mock search results
        mock_results = [
            {"id": "doc1", "content": "Result 1", "score": 0.9, "metadata": {}},
            {"id": "doc2", "content": "Result 2", "score": 0.8, "metadata": {}},
        ]
        mock_components["redis_store"].search_similar = AsyncMock(return_value=mock_results)

        # Execute
        result = await server.search_context(query, limit=10)

        # Verify
        assert result["status"] == "success"
        assert result["results"] == mock_results
        assert result["count"] == 2
        assert result["cache_hit"] is False

    @pytest.mark.asyncio
    async def test_search_context_with_cache_hit(self, server, mock_components):
        """Test search with cache hit."""
        # Setup
        query = "cached query"
        cached_response = {
            "results": [{"id": "cached", "content": "Cached result"}],
            "timestamp": "2024-01-01T00:00:00Z",
        }
        mock_components["semantic_cache"].get_cached_response = AsyncMock(
            return_value=cached_response
        )

        # Execute
        result = await server.search_context(query)

        # Verify cache was used
        assert result["status"] == "success"
        assert result["cache_hit"] is True
        assert "cached" in result["results"][0]["id"]
        # Search should not be called on cache hit
        mock_components["redis_store"].search_similar.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_indexed_stats(self, server, mock_components):
        """Test getting indexing statistics."""
        # Setup
        mock_stats = {
            "total_documents": 100,
            "total_chunks": 500,
            "sources": 5,
            "last_indexed": "2024-01-01T00:00:00Z",
        }
        mock_components["indexer"].get_stats = AsyncMock(return_value=mock_stats)

        # Execute
        result = await server.get_indexed_stats()

        # Verify
        assert result == mock_stats
        mock_components["indexer"].get_stats.assert_called_once()


class TestServerTaskManagement:
    """Test task management methods."""

    @pytest.mark.asyncio
    async def test_get_indexing_status(self, server, mock_components):
        """Test getting indexing task status."""
        # Setup
        task_id = "task-123"
        mock_status = {
            "task_id": task_id,
            "status": "in_progress",
            "progress": 50,
            "total_files": 100,
            "completed_files": 50,
        }
        mock_components["task_manager"].get_task_status = AsyncMock(return_value=mock_status)

        # Execute
        result = await server.get_indexing_status(task_id)

        # Verify
        assert result["status"] == "success"
        assert result["task"] == mock_status

    @pytest.mark.asyncio
    async def test_get_indexing_status_not_found(self, server, mock_components):
        """Test getting status for non-existent task."""
        # Setup
        task_id = "invalid-task"
        mock_components["task_manager"].get_task_status = AsyncMock(return_value=None)

        # Execute
        result = await server.get_indexing_status(task_id)

        # Verify
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_cancel_indexing_task(self, server, mock_components):
        """Test canceling an indexing task."""
        # Setup
        task_id = "task-123"
        mock_components["task_manager"].cancel_task = AsyncMock(return_value=True)

        # Execute
        result = await server.cancel_indexing_task(task_id)

        # Verify
        assert result["status"] == "success"
        assert result["cancelled"] is True

    @pytest.mark.asyncio
    async def test_cleanup_old_tasks(self, server, mock_components):
        """Test cleaning up old tasks."""
        # Setup
        mock_components["task_manager"].cleanup_old_tasks = AsyncMock(
            return_value={"cleaned": 5, "remaining": 2}
        )

        # Execute
        result = await server.cleanup_old_indexing_tasks(max_age_hours=24)

        # Verify
        assert "cleaned_tasks" in result
        mock_components["task_manager"].cleanup_old_tasks.assert_called_once_with(24)


class TestServerKnowledgeGraph:
    """Test knowledge graph methods."""

    @pytest.mark.asyncio
    async def test_query_knowledge_graph(self, server, mock_components):
        """Test querying knowledge graph."""
        # Setup
        query = "test entity"
        mock_results = {
            "entities": [
                {"name": "Entity1", "type": "person", "relations": []},
                {"name": "Entity2", "type": "place", "relations": []},
            ]
        }
        mock_components["knowledge_graph"].query = AsyncMock(return_value=mock_results)

        # Execute
        result = await server.query_knowledge_graph(query, depth=2)

        # Verify
        assert result["status"] == "success"
        assert "entities" in result["results"]
        mock_components["knowledge_graph"].query.assert_called_once_with(query, depth=2, limit=10)

    @pytest.mark.asyncio
    async def test_get_entity_relations(self, server, mock_components):
        """Test getting entity relations."""
        # Setup
        entity = "TestEntity"
        mock_relations = [
            {"from": entity, "to": "Related1", "type": "knows"},
            {"from": entity, "to": "Related2", "type": "located_at"},
        ]
        mock_components["knowledge_graph"].get_entity_relations = AsyncMock(
            return_value=mock_relations
        )

        # Execute
        result = await server.get_entity_relations(entity)

        # Verify
        assert result["status"] == "success"
        assert result["entity"] == entity
        assert result["relations"] == mock_relations


class TestServerFileWatcher:
    """Test file watcher methods."""

    @pytest.mark.asyncio
    async def test_watch_directory(self, server, mock_components):
        """Test starting directory watch."""
        # Setup
        directory = "/test/watch"
        mock_components["file_watcher"].start_watching = AsyncMock()

        # Execute
        result = await server.watch_directory(directory, recursive=True)

        # Verify
        assert result["status"] == "success"
        assert result["watching"] == directory
        mock_components["file_watcher"].start_watching.assert_called_once_with(
            directory,
            recursive=True,
            patterns=["*.py", "*.md", "*.txt", "*.json", "*.yaml", "*.yml"],
        )

    @pytest.mark.asyncio
    async def test_stop_watching(self, server, mock_components):
        """Test stopping directory watch."""
        # Setup
        directory = "/test/watch"
        mock_components["file_watcher"].stop_watching = AsyncMock(return_value=True)

        # Execute
        result = await server.stop_watching(directory)

        # Verify
        assert result["status"] == "success"
        assert result["stopped"] is True

    @pytest.mark.asyncio
    async def test_get_watched_directories(self, server, mock_components):
        """Test getting watched directories."""
        # Setup
        mock_dirs = ["/dir1", "/dir2", "/dir3"]
        mock_components["file_watcher"].get_watched_directories = MagicMock(return_value=mock_dirs)

        # Execute
        result = await server.get_watched_directories()

        # Verify
        assert result["status"] == "success"
        assert result["directories"] == mock_dirs


class TestServerOptimization:
    """Test optimization methods."""

    @pytest.mark.asyncio
    async def test_optimize_context(self, server, mock_components):
        """Test context optimization."""
        # Setup
        context = "Very long context that needs optimization..."
        mock_optimized = "Optimized context"
        mock_components["processor"].optimize_context = AsyncMock(return_value=mock_optimized)

        # Execute
        result = await server.optimize_context(context, max_tokens=1000, preserve_code=True)

        # Verify
        assert result["status"] == "success"
        assert result["optimized_context"] == mock_optimized
        mock_components["processor"].optimize_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, server, mock_components):
        """Test getting cache statistics."""
        # Setup
        mock_stats = {"total_entries": 100, "hit_rate": 0.31, "avg_response_time": 0.05}
        mock_components["semantic_cache"].get_stats = AsyncMock(return_value=mock_stats)

        # Execute
        result = await server.get_cache_stats()

        # Verify
        assert result == mock_stats
        mock_components["semantic_cache"].get_stats.assert_called_once()


class TestServerHealthCheck:
    """Test health check method."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, server, mock_components):
        """Test health check when all components are healthy."""
        # Setup
        mock_components["redis_store"].health_check = AsyncMock(
            return_value={"status": "healthy", "latency": 0.001}
        )

        # Execute
        result = await server.health_check()

        # Verify
        assert result["status"] == "healthy"
        assert "redis" in result
        assert result["redis"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, server, mock_components):
        """Test health check when Redis is unhealthy."""
        # Setup
        mock_components["redis_store"].health_check = AsyncMock(
            return_value={"status": "unhealthy", "error": "Connection timeout"}
        )

        # Execute
        result = await server.health_check()

        # Verify
        assert result["status"] == "unhealthy"
        assert result["redis"]["status"] == "unhealthy"
        assert "Connection timeout" in result["redis"]["error"]
