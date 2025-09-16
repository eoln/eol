"""Asynchronous task management for non-blocking indexing operations.

This module provides a task management system that allows the MCP server to:
- Start indexing operations immediately and return a task reference ID
- Track indexing progress in real-time without blocking
- Query task status, progress, and results
- Cancel running tasks gracefully
- Clean up completed/failed tasks automatically

The system is designed to ensure the AI agent connected to the MCP server
never gets blocked waiting for long-running indexing operations.

Example:
    Starting non-blocking indexing:

    >>> task_manager = AsyncTaskManager(redis_store)
    >>> task_id = await task_manager.start_indexing_task(
    ...     "/huge/repo",
    ...     indexer,
    ...     parallel_config
    ... )
    >>> print(f"Indexing started with task ID: {task_id}")

    Checking progress:

    >>> status = await task_manager.get_task_status(task_id)
    >>> print(f"Progress: {status.progress_percentage:.1f}% - {status.current_file}")

"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .parallel_indexer import IndexingCheckpoint, ParallelIndexer, ParallelIndexingConfig
from .redis_client import RedisVectorStore

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status enumeration."""

    PENDING = "pending"  # Task created but not started
    RUNNING = "running"  # Task currently executing
    COMPLETED = "completed"  # Task finished successfully
    FAILED = "failed"  # Task failed with error
    CANCELLED = "cancelled"  # Task was cancelled
    PAUSED = "paused"  # Task paused (for future use)


@dataclass
class IndexingTaskInfo:
    """Information about an indexing task.

    Contains all metadata needed to track and manage an indexing operation.
    """

    task_id: str
    status: TaskStatus
    folder_path: str
    source_id: str
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None

    # Progress tracking
    total_files: int = 0
    completed_files: int = 0
    failed_files: int = 0
    # current_file: Optional[str] = None  # Removed for simplicity

    # Results
    total_chunks: int = 0
    indexed_files: int = 0
    errors: list[str] = None
    result: dict[str, Any] | None = None  # IndexedSource as dict

    # Configuration
    recursive: bool = True
    force_reindex: bool = False
    parallel_config: dict[str, Any] | None = None

    # Error information
    error_message: str | None = None
    error_traceback: str | None = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def progress_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.completed_files / self.total_files) * 100.0

    @property
    def files_per_second(self) -> float:
        """Calculate processing rate."""
        if not self.started_at or self.completed_files == 0:
            return 0.0
        elapsed = time.time() - self.started_at
        return self.completed_files / elapsed if elapsed > 0 else 0.0

    @property
    def estimated_completion_time(self) -> float | None:
        """Estimate completion time in seconds."""
        if (
            self.status != TaskStatus.RUNNING
            or self.files_per_second <= 0
            or self.completed_files >= self.total_files
        ):
            return None

        remaining_files = self.total_files - self.completed_files
        return remaining_files / self.files_per_second

    @property
    def elapsed_time(self) -> float:
        """Total elapsed time for the task."""
        if not self.started_at:
            return 0.0
        end_time = self.completed_at or time.time()
        return end_time - self.started_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["status"] = self.status.value
        # Round numerical values to 2 decimal places
        data["progress_percentage"] = round(self.progress_percentage, 2)
        data["files_per_second"] = round(self.files_per_second, 2)
        data["estimated_completion_time"] = (
            round(self.estimated_completion_time, 2) if self.estimated_completion_time else None
        )
        data["elapsed_time"] = round(self.elapsed_time, 2)
        return data


class AsyncTaskManager:
    """Manager for asynchronous indexing tasks.

    Provides non-blocking task execution with progress tracking and status management.
    Tasks run in the background while the MCP server remains responsive.
    """

    def __init__(self, redis_store: RedisVectorStore):
        self.redis_store = redis_store
        self.running_tasks: dict[str, asyncio.Task] = {}
        self.task_info: dict[str, IndexingTaskInfo] = {}

        # Cleanup settings
        self.max_completed_tasks = 100
        self.task_ttl_hours = 24

    async def start_indexing_task(
        self,
        folder_path: Path | str,
        indexer: ParallelIndexer,
        source_id: str | None = None,
        recursive: bool = True,
        force_reindex: bool = False,
        parallel_config: ParallelIndexingConfig | None = None,
    ) -> str:
        """Start a new indexing task and return immediately with task ID.

        Args:
            folder_path: Directory to index
            indexer: ParallelIndexer instance to use
            source_id: Optional source identifier
            recursive: Whether to scan subdirectories
            force_reindex: Force reindexing of unchanged files
            parallel_config: Parallel processing configuration

        Returns:
            Unique task ID that can be used to track progress
        """
        folder_path = Path(folder_path).resolve()

        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Generate source ID if not provided
        if source_id is None:
            source_id = indexer._generate_source_id(folder_path)

        # Create task info
        task_info = IndexingTaskInfo(
            task_id=task_id,
            status=TaskStatus.PENDING,
            folder_path=str(folder_path),
            source_id=source_id,
            created_at=time.time(),
            recursive=recursive,
            force_reindex=force_reindex,
            parallel_config=asdict(parallel_config) if parallel_config else None,
        )

        self.task_info[task_id] = task_info

        # Start background task
        async_task = asyncio.create_task(
            self._execute_indexing_task(task_id, folder_path, indexer, parallel_config)
        )
        self.running_tasks[task_id] = async_task

        # Store task info in Redis for persistence
        await self._store_task_info(task_info)

        logger.info(f"Started indexing task {task_id} for {folder_path}")
        return task_id

    async def get_task_status(self, task_id: str) -> IndexingTaskInfo | None:
        """Get current status and progress of a task.

        Args:
            task_id: Task identifier returned by start_indexing_task

        Returns:
            IndexingTaskInfo with current status or None if task not found
        """
        # Try memory first
        if task_id in self.task_info:
            return self.task_info[task_id]

        # Fall back to Redis for persistence
        return await self._load_task_info(task_id)

    async def list_tasks(
        self, status_filter: TaskStatus | None = None, limit: int = 50
    ) -> list[IndexingTaskInfo]:
        """List tasks with optional status filtering.

        Args:
            status_filter: Only return tasks with this status
            limit: Maximum number of tasks to return

        Returns:
            List of IndexingTaskInfo objects
        """
        tasks = []

        # Get from memory (active tasks)
        for task_info in self.task_info.values():
            if status_filter is None or task_info.status == status_filter:
                tasks.append(task_info)

        # Get from Redis (completed/failed tasks)
        if len(tasks) < limit:
            redis_tasks = await self._load_all_task_info()
            for task_info in redis_tasks:
                if task_info.task_id not in self.task_info and (
                    status_filter is None or task_info.status == status_filter
                ):
                    tasks.append(task_info)

        # Sort by creation time (newest first)
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task.

        Args:
            task_id: Task identifier to cancel

        Returns:
            True if task was cancelled, False if not found or not cancellable
        """
        if task_id not in self.running_tasks:
            return False

        task = self.running_tasks[task_id]
        task_info = self.task_info.get(task_id)

        if task_info and task_info.status == TaskStatus.RUNNING:
            task.cancel()
            task_info.status = TaskStatus.CANCELLED
            task_info.completed_at = time.time()

            await self._store_task_info(task_info)

            logger.info(f"Cancelled indexing task {task_id}")
            return True

        return False

    async def cleanup_old_tasks(self) -> int:
        """Clean up old completed/failed tasks to free memory.

        Returns:
            Number of tasks cleaned up
        """
        current_time = time.time()
        cutoff_time = current_time - (self.task_ttl_hours * 3600)

        cleaned_count = 0

        # Clean up memory cache
        to_remove = []
        for task_id, task_info in self.task_info.items():
            if (
                task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
                and task_info.created_at < cutoff_time
            ):
                to_remove.append(task_id)

        for task_id in to_remove:
            del self.task_info[task_id]
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            cleaned_count += 1

        # Clean up Redis storage
        redis_cleaned = await self._cleanup_redis_tasks(cutoff_time)

        logger.info(f"Cleaned up {cleaned_count + redis_cleaned} old indexing tasks")
        return cleaned_count + redis_cleaned

    async def cleanup_stuck_tasks(self, timeout_minutes: int = 30) -> int:
        """Clean up stuck running tasks that have exceeded timeout.

        Tasks that have been running for longer than the timeout and show
        no progress are marked as failed and cleaned up.

        Args:
            timeout_minutes: Mark tasks as stuck after this many minutes (default 30)

        Returns:
            Number of stuck tasks cleaned up
        """
        timeout_seconds = timeout_minutes * 60
        current_time = time.time()
        stuck_count = 0

        # Check both memory and Redis for stuck tasks
        all_tasks = list(self.task_info.values())

        # Also load from Redis
        pattern = "indexing_task:*"
        cursor = 0
        while True:
            cursor, keys = await self.redis_store.async_redis.scan(cursor, match=pattern, count=100)
            for key in keys:
                task_id = (
                    key.split(":")[-1] if isinstance(key, str) else key.decode().split(":")[-1]
                )
                if task_id not in self.task_info:
                    task_info = await self._load_task_info(task_id)
                    if task_info:
                        all_tasks.append(task_info)
            if cursor == 0:
                break

        for task_info in all_tasks:
            # Check if task is stuck
            if task_info.status == TaskStatus.RUNNING and task_info.elapsed_time > timeout_seconds:

                # Check if there's any progress
                if task_info.completed_files == 0 or (
                    task_info.files_per_second < 0.01 and task_info.elapsed_time > 60
                ):

                    # Mark as failed
                    task_info.status = TaskStatus.FAILED
                    task_info.completed_at = current_time
                    task_info.error_message = (
                        f"Task timed out after {round(task_info.elapsed_time, 2)} "
                        "seconds with no progress"
                    )

                    # Store updated status
                    await self._store_task_info(task_info)

                    # Cancel if still in running_tasks
                    if task_info.task_id in self.running_tasks:
                        self.running_tasks[task_info.task_id].cancel()
                        del self.running_tasks[task_info.task_id]

                    stuck_count += 1
                    logger.warning(f"Cleaned up stuck task {task_info.task_id}")

        if stuck_count > 0:
            logger.info(f"Cleaned up {stuck_count} stuck running tasks")

        return stuck_count

    async def _execute_indexing_task(
        self,
        task_id: str,
        folder_path: Path,
        indexer: ParallelIndexer,
        parallel_config: ParallelIndexingConfig | None,
    ) -> None:
        """Execute indexing task in background."""
        task_info = self.task_info[task_id]

        try:
            # Update status to running
            task_info.status = TaskStatus.RUNNING
            task_info.started_at = time.time()
            await self._store_task_info(task_info)

            # Create progress callback
            def progress_callback(checkpoint: IndexingCheckpoint):
                """Update task progress from indexing checkpoint."""
                # Update task info with latest progress
                task_info.total_files = checkpoint.total_files
                task_info.completed_files = checkpoint.completed_files
                task_info.total_chunks = checkpoint.total_chunks
                # Update memory immediately for get_task_status
                self.task_info[task_id] = task_info
                # Store to Redis immediately for live stats (synchronous)
                asyncio.create_task(self._store_task_info(task_info))

            # Set parallel config if provided
            if parallel_config:
                indexer.parallel_config = parallel_config

            # Execute indexing
            result = await indexer.index_folder_parallel(
                folder_path,
                source_id=task_info.source_id,
                recursive=task_info.recursive,
                force_reindex=task_info.force_reindex,
                progress_callback=progress_callback,
            )

            # Task completed successfully
            task_info.status = TaskStatus.COMPLETED
            task_info.completed_at = time.time()
            task_info.result = asdict(result)
            task_info.indexed_files = result.indexed_files
            task_info.total_chunks = result.total_chunks

            logger.info(f"Indexing task {task_id} completed successfully")

        except asyncio.CancelledError:
            # Task was cancelled
            task_info.status = TaskStatus.CANCELLED
            task_info.completed_at = time.time()
            logger.info(f"Indexing task {task_id} was cancelled")

        except Exception as e:
            # Task failed
            task_info.status = TaskStatus.FAILED
            task_info.completed_at = time.time()
            task_info.error_message = str(e)

            import traceback

            task_info.error_traceback = traceback.format_exc()

            logger.error(f"Indexing task {task_id} failed: {e}")

        finally:
            # Clean up and persist final state
            await self._store_task_info(task_info)

            # Remove from active tasks
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

    async def _store_task_info(self, task_info: IndexingTaskInfo) -> None:
        """Store task info in Redis for persistence."""
        try:
            key = f"indexing_task:{task_info.task_id}"
            data = json.dumps(task_info.to_dict())

            await self.redis_store.async_redis.setex(
                key, self.task_ttl_hours * 3600, data  # TTL in seconds
            )
        except Exception as e:
            logger.error(f"Failed to store task info for {task_info.task_id}: {e}")

    async def _store_task_info_safe(self, task_info: IndexingTaskInfo) -> None:
        """Store task info in Redis with error handling (for background tasks)."""
        try:
            await self._store_task_info(task_info)
        except Exception as e:
            # Log but don't raise - this runs in background
            logger.warning(f"Background task info storage failed for {task_info.task_id}: {e}")
            # Continue without Redis storage - memory storage still works

    async def _load_task_info(self, task_id: str) -> IndexingTaskInfo | None:
        """Load task info from Redis."""
        try:
            key = f"indexing_task:{task_id}"
            data = await self.redis_store.async_redis.get(key)

            if data:
                task_dict = json.loads(data)
                # Convert status back from string
                task_dict["status"] = TaskStatus(task_dict["status"])

                # Create IndexingTaskInfo from dict (remove computed properties)
                computed_fields = [
                    "progress_percentage",
                    "files_per_second",
                    "estimated_completion_time",
                    "elapsed_time",
                ]
                for field in computed_fields:
                    task_dict.pop(field, None)

                return IndexingTaskInfo(**task_dict)

        except Exception as e:
            logger.error(f"Failed to load task info for {task_id}: {e}")

        return None

    async def _load_all_task_info(self) -> list[IndexingTaskInfo]:
        """Load all task info from Redis."""
        tasks = []
        try:
            keys = await self.redis_store.async_redis.keys("indexing_task:*")

            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode()

                task_id = key.split(":", 1)[1]
                task_info = await self._load_task_info(task_id)

                if task_info:
                    tasks.append(task_info)

        except Exception as e:
            logger.error(f"Failed to load all task info: {e}")

        return tasks

    async def _cleanup_redis_tasks(self, cutoff_time: float) -> int:
        """Clean up old tasks from Redis."""
        cleaned = 0
        try:
            keys = await self.redis_store.async_redis.keys("indexing_task:*")

            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode()

                # Get task creation time
                data = await self.redis_store.async_redis.get(key)
                if data:
                    task_dict = json.loads(data)
                    created_at = task_dict.get("created_at", 0)

                    if created_at < cutoff_time:
                        await self.redis_store.async_redis.delete(key)
                        cleaned += 1

        except Exception as e:
            logger.error(f"Failed to cleanup Redis tasks: {e}")

        return cleaned
