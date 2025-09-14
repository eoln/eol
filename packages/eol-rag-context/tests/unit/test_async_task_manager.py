"""Unit tests for AsyncTaskManager - non-blocking indexing task management."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from eol.rag_context.async_task_manager import AsyncTaskManager, IndexingTaskInfo, TaskStatus


class TestAsyncTaskManager:
    """Test AsyncTaskManager functionality."""

    @pytest.fixture
    def mock_redis_store(self):
        """Mock RedisVectorStore."""
        mock_store = MagicMock()
        mock_store.redis = AsyncMock()
        return mock_store

    @pytest.fixture
    def mock_parallel_indexer(self):
        """Mock ParallelIndexer."""
        mock_indexer = AsyncMock()
        mock_indexer._generate_source_id.return_value = "test-source-123"
        return mock_indexer

    @pytest.fixture
    def task_manager(self, mock_redis_store):
        """Create AsyncTaskManager instance."""
        return AsyncTaskManager(mock_redis_store)

    @pytest.mark.asyncio
    async def test_start_indexing_task(self, task_manager, mock_parallel_indexer):
        """Test starting an indexing task."""
        # Mock the parallel indexing result
        mock_result = MagicMock()
        mock_result.source_id = "test-source-123"
        mock_result.indexed_files = 10
        mock_result.total_chunks = 50
        mock_parallel_indexer.index_folder_parallel = AsyncMock(return_value=mock_result)

        task_id = await task_manager.start_indexing_task(
            Path("/test/path"), mock_parallel_indexer, recursive=True, force_reindex=False
        )

        assert task_id is not None
        assert len(task_id) == 36  # UUID4 length
        assert task_id in task_manager.task_info

        task_info = task_manager.task_info[task_id]
        assert task_info.status == TaskStatus.PENDING
        assert task_info.folder_path == "/test/path"
        assert task_info.recursive is True

    @pytest.mark.asyncio
    async def test_get_task_status(self, task_manager):
        """Test getting task status."""
        # Test non-existent task
        status = await task_manager.get_task_status("non-existent")
        assert status is None

        # Create a task info manually
        task_info = IndexingTaskInfo(
            task_id="test-task-123",
            status=TaskStatus.RUNNING,
            folder_path="/test/path",
            source_id="test-source",
            created_at=time.time(),
            total_files=100,
            completed_files=25,
        )
        task_manager.task_info["test-task-123"] = task_info

        # Get task status
        status = await task_manager.get_task_status("test-task-123")
        assert status is not None
        assert status.task_id == "test-task-123"
        assert status.progress_percentage == 25.0

    @pytest.mark.asyncio
    async def test_list_tasks(self, task_manager):
        """Test listing tasks."""
        # Test empty task list
        tasks = await task_manager.list_tasks()
        assert tasks == []

        # Add some tasks
        task1 = IndexingTaskInfo(
            task_id="task-1",
            status=TaskStatus.RUNNING,
            folder_path="/path1",
            source_id="source-1",
            created_at=time.time(),
        )
        task2 = IndexingTaskInfo(
            task_id="task-2",
            status=TaskStatus.COMPLETED,
            folder_path="/path2",
            source_id="source-2",
            created_at=time.time() - 100,
        )

        task_manager.task_info["task-1"] = task1
        task_manager.task_info["task-2"] = task2

        # List all tasks
        tasks = await task_manager.list_tasks()
        assert len(tasks) == 2

        # List filtered tasks
        running_tasks = await task_manager.list_tasks(status_filter=TaskStatus.RUNNING)
        assert len(running_tasks) == 1
        assert running_tasks[0].task_id == "task-1"

    @pytest.mark.asyncio
    async def test_cancel_task(self, task_manager):
        """Test cancelling a task."""
        # Test cancelling non-existent task
        success = await task_manager.cancel_task("non-existent")
        assert success is False

        # Create a running task
        mock_asyncio_task = AsyncMock()
        task_info = IndexingTaskInfo(
            task_id="test-task",
            status=TaskStatus.RUNNING,
            folder_path="/test",
            source_id="test-source",
            created_at=time.time(),
        )

        task_manager.running_tasks["test-task"] = mock_asyncio_task
        task_manager.task_info["test-task"] = task_info

        # Cancel the task
        success = await task_manager.cancel_task("test-task")
        assert success is True
        mock_asyncio_task.cancel.assert_called_once()
        assert task_info.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cleanup_old_tasks(self, task_manager):
        """Test cleaning up old tasks."""
        current_time = time.time()
        old_time = current_time - (25 * 3600)  # 25 hours ago

        # Add old completed task
        old_task = IndexingTaskInfo(
            task_id="old-task",
            status=TaskStatus.COMPLETED,
            folder_path="/old",
            source_id="old-source",
            created_at=old_time,
        )

        # Add recent running task
        recent_task = IndexingTaskInfo(
            task_id="recent-task",
            status=TaskStatus.RUNNING,
            folder_path="/recent",
            source_id="recent-source",
            created_at=current_time,
        )

        task_manager.task_info["old-task"] = old_task
        task_manager.task_info["recent-task"] = recent_task

        # Mock Redis cleanup
        with patch.object(task_manager, "_cleanup_redis_tasks", return_value=2):
            cleaned = await task_manager.cleanup_old_tasks()

        # Old task should be removed, recent task should remain
        assert cleaned >= 1  # At least the old task
        assert "old-task" not in task_manager.task_info
        assert "recent-task" in task_manager.task_info

    def test_task_info_properties(self):
        """Test IndexingTaskInfo computed properties."""
        task_info = IndexingTaskInfo(
            task_id="test",
            status=TaskStatus.RUNNING,
            folder_path="/test",
            source_id="test-source",
            created_at=time.time(),
            started_at=time.time() - 60,  # Started 60 seconds ago
            total_files=100,
            completed_files=40,
        )

        # Test progress percentage
        assert task_info.progress_percentage == 40.0

        # Test files per second (should be roughly 40/60 = 0.67)
        assert task_info.files_per_second > 0.5
        assert task_info.files_per_second < 1.0

        # Test elapsed time
        assert task_info.elapsed_time > 55  # Should be around 60 seconds
        assert task_info.elapsed_time < 65

        # Test estimated completion time
        estimated = task_info.estimated_completion_time
        assert estimated is not None
        assert estimated > 80  # Should take more time for remaining files
        assert estimated < 100

    def test_task_info_edge_cases(self):
        """Test IndexingTaskInfo edge cases."""
        # Task with no files
        task_info = IndexingTaskInfo(
            task_id="test",
            status=TaskStatus.RUNNING,
            folder_path="/test",
            source_id="test-source",
            created_at=time.time(),
            total_files=0,
            completed_files=0,
        )

        assert task_info.progress_percentage == 0.0
        assert task_info.files_per_second == 0.0
        assert task_info.estimated_completion_time is None

        # Task not started yet
        task_info_not_started = IndexingTaskInfo(
            task_id="test2",
            status=TaskStatus.PENDING,
            folder_path="/test2",
            source_id="test-source-2",
            created_at=time.time(),
            started_at=None,
            total_files=100,
            completed_files=0,
        )

        assert task_info_not_started.files_per_second == 0.0
        assert task_info_not_started.elapsed_time == 0.0
        assert task_info_not_started.estimated_completion_time is None
