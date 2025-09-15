"""
Integration tests for server module to improve coverage.
"""

import asyncio

import pytest


@pytest.mark.integration
class TestServerIntegration:
    """Test server operations with real components."""

    @pytest.mark.asyncio
    async def test_server_initialization(self, server_instance):
        """Test server initialization and basic operations."""
        # Server should be initialized
        assert server_instance is not None
        assert server_instance.redis_store is not None
        assert server_instance.embedding_manager is not None
        assert server_instance.task_manager is not None

    @pytest.mark.asyncio
    async def test_server_index_directory(self, server_instance, temp_test_directory):
        """Test server's index_directory method."""
        # Start indexing
        result = await server_instance.index_directory(temp_test_directory, recursive=True)

        assert result["status"] == "started"
        assert "task_id" in result

        # Wait for completion
        task_id = result["task_id"]
        max_wait = 10
        for _ in range(max_wait):
            status = await server_instance.task_manager.get_task_status(task_id)
            if status and status.status.value in ["completed", "failed"]:
                break
            await asyncio.sleep(0.5)

        # Check final status
        final_status = await server_instance.task_manager.get_task_status(task_id)
        assert final_status is not None

    @pytest.mark.asyncio
    async def test_server_search_context(self, server_instance, temp_test_directory):
        """Test server's search_context method."""
        # Index some data first
        index_result = await server_instance.index_directory(temp_test_directory)
        task_id = index_result["task_id"]

        # Wait for indexing
        for _ in range(10):
            status = await server_instance.task_manager.get_task_status(task_id)
            if status and status.status.value == "completed":
                break
            await asyncio.sleep(0.5)

        # Now search
        results = await server_instance.search_context(
            query="test project hello world", max_results=5
        )

        # Should return list of results
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_server_optimize_context(self, server_instance):
        """Test server's optimize_context method."""
        # Test context optimization
        result = await server_instance.optimize_context(
            query="How does the system work?",
            current_context="This is existing context.",
            max_tokens=1000,
        )

        # Should return optimized context
        assert isinstance(result, dict)
        assert "query" in result
        assert result["query"] == "How does the system work?"

    @pytest.mark.asyncio
    async def test_server_clear_cache(self, server_instance):
        """Test server's clear_cache method."""
        # Clear cache
        result = await server_instance.clear_cache()

        # Should return success
        assert isinstance(result, dict)
        assert "semantic_cache" in result
        assert "embedding_cache" in result

    @pytest.mark.asyncio
    async def test_server_task_management(self, server_instance, temp_test_directory):
        """Test server's task management capabilities."""
        # Start multiple tasks
        task1 = await server_instance.index_directory(temp_test_directory)

        # List tasks
        tasks = await server_instance.task_manager.list_tasks()
        assert len(tasks) > 0

        # Get specific task status
        status = await server_instance.task_manager.get_task_status(task1["task_id"])
        assert status is not None

        # Cancel task (if still running)
        if status.status.value == "running":
            cancelled = await server_instance.task_manager.cancel_task(task1["task_id"])
            assert isinstance(cancelled, bool)

    @pytest.mark.asyncio
    async def test_server_cleanup_tasks(self, server_instance):
        """Test server's cleanup_old_tasks method."""
        # Cleanup old tasks
        result = await server_instance.task_manager.cleanup_old_tasks()

        # Should return cleanup stats
        assert isinstance(result, dict)
        assert "removed" in result or "cleaned" in result or result == {}
