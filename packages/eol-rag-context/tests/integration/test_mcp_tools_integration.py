"""
Integration tests for MCP tool endpoints.
Tests the actual MCP tools provided by the server.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from eol.rag_context.server import SearchContextRequest, WatchDirectoryRequest


@pytest.mark.integration
class TestMCPToolsIntegration:
    """Test MCP tool endpoints with real server."""

    @pytest.mark.asyncio
    async def test_start_indexing_tool(self, server_instance):
        """Test start_indexing MCP tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Create test file
            (test_dir / "test.py").write_text(
                """
def test_function():
    '''Test function for MCP tool testing.'''
    return "Hello from MCP tool test"
"""
            )

            # Simulate tool call
            result = await server_instance.index_directory(test_dir, recursive=True)

            # Should return task started response
            assert result["status"] == "started"
            assert "task_id" in result
            assert str(result["path"]) == str(test_dir)

            # Wait for task completion
            task_id = result["task_id"]
            max_wait = 30
            wait_time = 0
            while wait_time < max_wait:
                status = await server_instance.task_manager.get_task_status(task_id)
                if status and status.status.value in ["completed", "failed"]:
                    break
                await asyncio.sleep(1)
                wait_time += 1

            # Verify task completed
            final_status = await server_instance.task_manager.get_task_status(task_id)
            assert final_status is not None
            assert final_status.status.value == "completed"
            assert final_status.total_files > 0

    @pytest.mark.asyncio
    async def test_get_indexing_status_tool(self, server_instance):
        """Test get_indexing_status MCP tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Create test file
            (test_dir / "status_test.py").write_text("# Test file for status checking")

            # Start indexing
            result = await server_instance.index_directory(test_dir, recursive=True)
            task_id = result["task_id"]

            # Test getting status
            status = await server_instance.task_manager.get_task_status(task_id)
            assert status is not None
            assert hasattr(status, "task_id")
            assert status.task_id == task_id
            assert hasattr(status, "status")

            # Wait for completion to test completed status
            max_wait = 30
            wait_time = 0
            while wait_time < max_wait:
                status = await server_instance.task_manager.get_task_status(task_id)
                if status and status.status.value in ["completed", "failed"]:
                    break
                await asyncio.sleep(1)
                wait_time += 1

            # Test final status
            final_status = await server_instance.task_manager.get_task_status(task_id)
            assert final_status.status.value == "completed"
            assert final_status.progress_percentage == 100.0

    @pytest.mark.asyncio
    async def test_list_indexing_tasks_tool(self, server_instance):
        """Test list_indexing_tasks MCP tool."""
        # Get initial task count
        initial_tasks = await server_instance.task_manager.list_tasks()
        initial_count = len(initial_tasks)

        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            (test_dir / "list_test.py").write_text("# Test file for list tasks")

            # Start a task
            result = await server_instance.index_directory(test_dir, recursive=True)
            task_id = result["task_id"]

            # List tasks - should include our new task
            tasks = await server_instance.task_manager.list_tasks()
            assert len(tasks) == initial_count + 1

            # Find our task
            our_task = None
            for task in tasks:
                if task.task_id == task_id:
                    our_task = task
                    break

            assert our_task is not None
            assert our_task.folder_path == str(test_dir)

    @pytest.mark.asyncio
    async def test_cancel_indexing_task_tool(self, server_instance):
        """Test cancel_indexing_task MCP tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Create multiple files to have a longer-running task
            for i in range(5):
                (test_dir / f"cancel_test_{i}.py").write_text(f"# Test file {i} for cancellation")

            # Start indexing
            result = await server_instance.index_directory(test_dir, recursive=True)
            task_id = result["task_id"]

            # Try to cancel immediately (task might complete before cancellation)
            cancelled = await server_instance.task_manager.cancel_task(task_id)

            # Cancellation should succeed (returns True) even if task already completed
            assert cancelled is True

            # Check final status
            final_status = await server_instance.task_manager.get_task_status(task_id)
            assert final_status is not None
            # Status should be either cancelled or completed depending on timing
            assert final_status.status.value in ["cancelled", "completed"]

    @pytest.mark.asyncio
    async def test_cleanup_old_indexing_tasks_tool(self, server_instance):
        """Test cleanup_old_indexing_tasks MCP tool."""
        # Create and complete some tasks first
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            (test_dir / "cleanup_test.py").write_text("# Test file for cleanup")

            # Start and wait for completion
            result = await server_instance.index_directory(test_dir, recursive=True)
            task_id = result["task_id"]

            # Wait for completion
            max_wait = 30
            wait_time = 0
            while wait_time < max_wait:
                status = await server_instance.task_manager.get_task_status(task_id)
                if status and status.status.value in ["completed", "failed"]:
                    break
                await asyncio.sleep(1)
                wait_time += 1

            # Now test cleanup
            cleaned_count = await server_instance.task_manager.cleanup_old_tasks()

            # Should return the number of tasks cleaned (≥ 0)
            assert isinstance(cleaned_count, int)
            assert cleaned_count >= 0

    @pytest.mark.asyncio
    async def test_search_context_workflow(self, server_instance, temp_test_directory):
        """Test search context after indexing."""
        # First index some content
        result = await server_instance.index_directory(temp_test_directory, recursive=True)
        task_id = result["task_id"]

        # Wait for indexing to complete
        max_wait = 30
        wait_time = 0
        while wait_time < max_wait:
            status = await server_instance.task_manager.get_task_status(task_id)
            if status and status.status.value in ["completed", "failed"]:
                break
            await asyncio.sleep(1)
            wait_time += 1

        final_status = await server_instance.task_manager.get_task_status(task_id)
        assert final_status.status.value == "completed"

        # Now test context search
        search_request = SearchContextRequest(
            query="hello world function", max_results=5, min_relevance=0.7
        )

        # Get query embedding
        query_embedding = await server_instance.embedding_manager.get_embedding(
            search_request.query
        )

        # Search via redis store
        results = await server_instance.redis_store.vector_search(
            query_embedding=query_embedding, hierarchy_level=3, k=search_request.max_results
        )

        # Results should be a list (may be empty if no good matches)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_watch_directory_workflow(self, server_instance):
        """Test watch directory MCP tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Create initial file
            (test_dir / "watch_test.py").write_text("# Initial file for watching")

            # Test watch directory
            watch_request = WatchDirectoryRequest(
                path=str(test_dir), file_patterns=["*.py", "*.md"], recursive=True
            )

            # Call watch directory method
            result = await server_instance.watch_directory(
                test_dir, patterns=watch_request.file_patterns
            )

            # Should return some result (implementation dependent)
            assert result is not None

    @pytest.mark.asyncio
    async def test_concurrent_mcp_operations(self, server_instance):
        """Test concurrent MCP tool operations."""
        import shutil
        import tempfile

        # Create multiple temporary directories
        temp_dirs = []
        for i in range(3):
            tmpdir = tempfile.mkdtemp()
            test_dir = Path(tmpdir)
            (test_dir / f"concurrent_{i}.py").write_text(f"# Concurrent test file {i}")
            temp_dirs.append(test_dir)

        try:
            # Start multiple indexing operations concurrently
            tasks = []
            for test_dir in temp_dirs:
                task = server_instance.index_directory(test_dir, recursive=True)
                tasks.append(task)

            # Gather all results
            results = await asyncio.gather(*tasks)

            # All should start successfully
            assert len(results) == 3
            for result in results:
                assert result["status"] == "started"
                assert "task_id" in result

            # Wait for all to complete
            task_ids = [r["task_id"] for r in results]
            max_wait = 60  # Longer wait for concurrent operations

            for task_id in task_ids:
                wait_time = 0
                while wait_time < max_wait:
                    status = await server_instance.task_manager.get_task_status(task_id)
                    if status and status.status.value in ["completed", "failed"]:
                        break
                    await asyncio.sleep(1)
                    wait_time += 1

                final_status = await server_instance.task_manager.get_task_status(task_id)
                assert final_status is not None
                assert final_status.status.value == "completed"

        finally:
            # Clean up temporary directories
            for temp_dir in temp_dirs:
                shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_error_handling_in_mcp_tools(self, server_instance):
        """Test error handling in MCP tools."""
        # Test with non-existent directory - in non-blocking mode, this starts a task
        # that fails later rather than raising immediately
        nonexistent_dir = Path("/nonexistent/directory/path")

        # Start indexing non-existent directory
        result = await server_instance.index_directory(nonexistent_dir, recursive=True)

        # Should start task but fail later
        assert result["status"] == "started"
        task_id = result["task_id"]

        # Wait for task to fail
        max_wait = 10
        wait_time = 0
        while wait_time < max_wait:
            status = await server_instance.task_manager.get_task_status(task_id)
            if status and status.status.value in ["failed", "completed"]:
                break
            await asyncio.sleep(1)
            wait_time += 1

        # Task should have failed
        final_status = await server_instance.task_manager.get_task_status(task_id)
        assert final_status is not None
        assert final_status.status.value == "failed"

        # Test getting status for non-existent task
        nonexistent_task_id = "nonexistent-task-id-123"
        status = await server_instance.task_manager.get_task_status(nonexistent_task_id)
        assert status is None  # Should return None for non-existent task

    @pytest.mark.asyncio
    async def test_mcp_tool_parameter_validation(self, server_instance):
        """Test parameter validation in MCP tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            (test_dir / "param_test.py").write_text("# Parameter validation test")

            # Test with valid parameters
            result = await server_instance.index_directory(test_dir, recursive=True)
            assert result["status"] == "started"

            # Test task listing (no parameters required)
            tasks = await server_instance.task_manager.list_tasks()
            assert isinstance(tasks, list)
