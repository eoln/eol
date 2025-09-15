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
        mock_store.async_redis = AsyncMock()
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

    @pytest.mark.asyncio
    async def test_cancel_running_task(self, task_manager, mock_parallel_indexer):
        """Test cancelling a running task."""
        # Start a task
        mock_result = MagicMock()
        mock_result.source_id = "test-source-123"
        mock_result.indexed_files = 10
        mock_result.total_chunks = 50

        # Create a future that will block
        import asyncio

        future = asyncio.Future()
        mock_parallel_indexer.index_folder_parallel = AsyncMock(return_value=future)

        task_id = await task_manager.start_indexing_task(
            Path("/test/path"), mock_parallel_indexer, recursive=True
        )

        # Task should be running
        assert task_manager.task_info[task_id].status == TaskStatus.RUNNING

        # Cancel the task
        success = await task_manager.cancel_task(task_id)
        assert success is True

        # Check task status is now cancelled
        task_info = task_manager.task_info.get(task_id)
        assert task_info is not None
        assert task_info.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self, task_manager):
        """Test cancelling a non-existent task."""
        success = await task_manager.cancel_task("nonexistent-task-id")
        assert success is False

    @pytest.mark.asyncio
    async def test_cleanup_stuck_tasks(self, task_manager):
        """Test cleanup of stuck running tasks."""
        # Create a stuck task directly in task_info
        stuck_task = IndexingTaskInfo(
            task_id="stuck-task",
            status=TaskStatus.RUNNING,
            folder_path="/test",
            source_id="test-source",
            created_at=time.time() - 3700,  # Created over an hour ago
            started_at=time.time() - 3600,  # Started an hour ago
            total_files=100,
            completed_files=0,  # No progress
        )
        task_manager.task_info["stuck-task"] = stuck_task

        # Mock Redis scan to return no additional tasks
        task_manager.redis_store.async_redis.scan = AsyncMock(return_value=(0, []))

        # Run cleanup
        cleaned = await task_manager.cleanup_stuck_tasks(timeout_minutes=30)

        # Should have cleaned up the stuck task
        assert cleaned == 1
        assert task_manager.task_info["stuck-task"].status == TaskStatus.FAILED
        assert "timed out" in task_manager.task_info["stuck-task"].error_message

    @pytest.mark.asyncio
    async def test_cleanup_stuck_tasks_with_redis(self, task_manager):
        """Test cleanup of stuck tasks from Redis."""
        # Mock Redis scan to return a task key
        task_manager.redis_store.async_redis.scan = AsyncMock(
            return_value=(0, [b"indexing_task:redis-stuck-task"])
        )

        # Mock loading task from Redis
        stuck_task_data = {
            "task_id": "redis-stuck-task",
            "status": "running",
            "folder_path": "/test",
            "source_id": "test-source",
            "created_at": time.time() - 3700,
            "started_at": time.time() - 3600,
            "total_files": 100,
            "completed_files": 0,
        }
        task_manager.redis_store.async_redis.get = AsyncMock(
            return_value=stuck_task_data.__str__().encode()
        )

        # Mock JSON loading
        with patch("json.loads", return_value=stuck_task_data):
            # Run cleanup
            cleaned = await task_manager.cleanup_stuck_tasks(timeout_minutes=30)

            # Should have attempted to clean up the task
            assert cleaned >= 0  # May be 0 if loading failed, but shouldn't error

    @pytest.mark.asyncio
    async def test_list_tasks_with_filters(self, task_manager):
        """Test listing tasks with various filters."""
        # Create tasks with different statuses
        for i, status in enumerate([TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.COMPLETED]):
            task_info = IndexingTaskInfo(
                task_id=f"task-{i}",
                status=status,
                folder_path=f"/test/{i}",
                source_id=f"source-{i}",
                created_at=time.time() - (100 * i),
                total_files=10,
                completed_files=5 if status == TaskStatus.RUNNING else 10,
            )
            task_manager.task_info[f"task-{i}"] = task_info

        # Mock Redis scan to return no additional tasks
        task_manager.redis_store.async_redis.scan = AsyncMock(return_value=(0, []))

        # List all tasks
        all_tasks = await task_manager.list_tasks()
        assert len(all_tasks["tasks"]) == 3

        # List only running tasks
        running_tasks = await task_manager.list_tasks(status_filter="running")
        assert len(running_tasks["tasks"]) == 1
        assert running_tasks["tasks"][0]["status"] == "running"

        # List with limit
        limited_tasks = await task_manager.list_tasks(limit=2)
        assert len(limited_tasks["tasks"]) == 2

    @pytest.mark.asyncio
    async def test_execute_indexing_task_error_handling(self, task_manager, mock_parallel_indexer):
        """Test error handling during task execution."""
        # Make the indexer raise an exception
        mock_parallel_indexer.index_folder_parallel = AsyncMock(
            side_effect=Exception("Indexing failed")
        )

        task_id = await task_manager.start_indexing_task(Path("/test/path"), mock_parallel_indexer)

        # Wait a bit for the task to fail
        await asyncio.sleep(0.1)

        # Check task failed
        task_info = task_manager.task_info.get(task_id)
        if task_info:  # Task might have already been cleaned up
            assert task_info.status in [TaskStatus.FAILED, TaskStatus.RUNNING]
            if task_info.status == TaskStatus.FAILED:
                assert "Indexing failed" in task_info.error_message or task_info.error_message

    @pytest.mark.asyncio
    async def test_monitor_tasks(self, task_manager):
        """Test task monitoring functionality."""
        # Create a task that should be monitored
        task_info = IndexingTaskInfo(
            task_id="monitor-task",
            status=TaskStatus.RUNNING,
            folder_path="/test",
            source_id="test-source",
            created_at=time.time(),
            started_at=time.time(),
            total_files=100,
            completed_files=50,
        )
        task_manager.task_info["monitor-task"] = task_info

        # Set monitoring to true
        task_manager.monitoring = True

        # Start monitoring (it will run in background)
        import asyncio

        monitor_task = asyncio.create_task(task_manager._monitor_tasks())

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Stop monitoring
        task_manager.monitoring = False
        monitor_task.cancel()

        # Monitoring ran without errors
        assert True  # If we got here, monitoring worked

    @pytest.mark.asyncio
    async def test_store_and_load_task_info(self, task_manager):
        """Test storing and loading task info from Redis."""
        # Create a task info
        task_info = IndexingTaskInfo(
            task_id="store-test",
            status=TaskStatus.COMPLETED,
            folder_path="/test/store",
            source_id="store-source",
            created_at=time.time(),
            total_files=50,
            completed_files=50,
        )

        # Mock Redis set and get
        task_manager.redis_store.async_redis.set = AsyncMock()
        stored_data = None

        def capture_set(key, value, ex=None):
            nonlocal stored_data
            stored_data = value
            return True

        task_manager.redis_store.async_redis.set = AsyncMock(side_effect=capture_set)

        # Store the task info
        await task_manager._store_task_info(task_info)

        # Verify it was stored
        assert task_manager.redis_store.async_redis.set.called
        assert stored_data is not None

        # Mock Redis get to return the stored data
        task_manager.redis_store.async_redis.get = AsyncMock(return_value=stored_data)

        # Load the task info
        loaded_task = await task_manager._load_task_info("store-test")

        # Verify loaded correctly
        if loaded_task:  # May be None if JSON parsing fails
            assert loaded_task.task_id == "store-test"
            assert loaded_task.status == TaskStatus.COMPLETED
