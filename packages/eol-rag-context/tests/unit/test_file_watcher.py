"""Focused tests for file_watcher.py module.

This test file contains meaningful tests for the file watching components,
extracted from coverage booster files and enhanced with real functionality testing.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import separately to avoid mocking issues
from eol.rag_context.file_watcher import (
    ChangeType,
    FileChange,
    FileChangeHandler,
    FileWatcher,
    WatchedSource,
)


class TestChangeType:
    """Test ChangeType enum."""

    def test_change_type_values(self):
        """Test ChangeType enum values."""
        assert ChangeType.CREATED.value == "created"
        assert ChangeType.MODIFIED.value == "modified"
        assert ChangeType.DELETED.value == "deleted"
        assert ChangeType.MOVED.value == "moved"

    def test_change_type_membership(self):
        """Test ChangeType membership checks."""
        creation_types = {ChangeType.CREATED, ChangeType.MODIFIED}
        assert ChangeType.CREATED in creation_types
        assert ChangeType.DELETED not in creation_types


class TestFileChange:
    """Test FileChange dataclass."""

    def test_file_change_creation(self):
        """Test FileChange creation with all fields."""
        old_path = Path("/old/path/file.py")
        new_path = Path("/new/path/file.py")
        timestamp = time.time()

        change = FileChange(
            path=new_path,
            change_type=ChangeType.MOVED,
            timestamp=timestamp,
            old_path=old_path,
            metadata={"source_id": "test_src", "reason": "user_action"},
        )

        assert change.path == new_path
        assert change.change_type == ChangeType.MOVED
        assert change.timestamp == timestamp
        assert change.old_path == old_path
        assert change.metadata["source_id"] == "test_src"

    def test_file_change_defaults(self):
        """Test FileChange with default values."""
        path = Path("/test/file.py")

        change = FileChange(path=path, change_type=ChangeType.CREATED)

        assert isinstance(change.timestamp, float)
        assert change.old_path is None
        assert change.metadata == {}


class TestWatchedSource:
    """Test WatchedSource dataclass."""

    def test_watched_source_creation(self):
        """Test WatchedSource creation."""
        path = Path("/project/src")

        source = WatchedSource(
            path=path,
            source_id="src_123",
            recursive=True,
            file_patterns=["*.py", "*.pyi"],
            change_count=5,
        )

        assert source.path == path
        assert source.source_id == "src_123"
        assert source.recursive is True
        assert source.file_patterns == ["*.py", "*.pyi"]
        assert source.change_count == 5

    def test_watched_source_defaults(self):
        """Test WatchedSource with default values."""
        path = Path("/project")

        source = WatchedSource(path=path, source_id="proj_456")

        assert source.recursive is True
        assert source.file_patterns == []
        assert source.change_count == 0
        assert isinstance(source.last_scan, float)


class TestFileChangeHandler:
    """Test FileChangeHandler class."""

    @pytest.fixture
    def mock_watcher(self):
        """Create a mock FileWatcher."""
        watcher = MagicMock()
        watcher._handle_change = AsyncMock()
        return watcher

    @pytest.fixture
    def mock_scanner(self):
        """Create a mock FolderScanner."""
        scanner = MagicMock()
        scanner._should_ignore = MagicMock(return_value=False)
        return scanner

    @pytest.fixture
    def handler(self, mock_watcher):
        """Create FileChangeHandler instance."""
        source_path = Path("/test/project")
        source_id = "test_src"
        file_patterns = ["*.py", "*.md"]

        # Mock the FolderScanner class completely
        mock_scanner = MagicMock()
        mock_scanner._should_ignore.return_value = False

        with patch("eol.rag_context.file_watcher.FolderScanner", return_value=mock_scanner):
            handler = FileChangeHandler(mock_watcher, source_path, source_id, file_patterns)

        return handler

    def test_handler_initialization(self, handler, mock_watcher):
        """Test FileChangeHandler initialization."""
        assert handler.watcher == mock_watcher
        assert handler.source_path == Path("/test/project")
        assert handler.source_id == "test_src"
        assert handler.file_patterns == ["*.py", "*.md"]

    def test_should_process_valid_file(self, handler):
        """Test _should_process with valid file."""
        with patch("pathlib.Path.is_dir", return_value=False):
            with patch("pathlib.Path.match", side_effect=lambda pattern: pattern == "*.py"):
                result = handler._should_process("/test/file.py")
                assert result is True

    def test_should_process_directory(self, handler):
        """Test _should_process with directory."""
        with patch("pathlib.Path.is_dir", return_value=True):
            result = handler._should_process("/test/directory")
            assert result is False

    def test_should_process_ignored_file(self, handler):
        """Test _should_process with ignored file."""
        handler.scanner._should_ignore.return_value = True

        with patch("pathlib.Path.is_dir", return_value=False):
            result = handler._should_process("/test/file.pyc")
            assert result is False

    def test_should_process_no_patterns(self, handler):
        """Test _should_process with no file patterns."""
        handler.file_patterns = []

        with patch("pathlib.Path.is_dir", return_value=False):
            result = handler._should_process("/test/any_file.txt")
            assert result is True

    def test_on_created_valid_file(self, handler, mock_watcher):
        """Test on_created event with valid file."""
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/new_file.py"

        with patch.object(handler, "_should_process", return_value=True):
            with patch("asyncio.create_task") as mock_create_task:
                handler.on_created(event)

                # Verify task was created
                mock_create_task.assert_called_once()

    def test_on_created_directory(self, handler, mock_watcher):
        """Test on_created event with directory."""
        event = MagicMock()
        event.is_directory = True
        event.src_path = "/test/new_directory"

        with patch("asyncio.create_task") as mock_create_task:
            handler.on_created(event)

            # Should not create task for directories
            mock_create_task.assert_not_called()

    def test_on_modified_valid_file(self, handler, mock_watcher):
        """Test on_modified event with valid file."""
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/modified_file.py"

        with patch.object(handler, "_should_process", return_value=True):
            with patch("asyncio.create_task") as mock_create_task:
                handler.on_modified(event)

                mock_create_task.assert_called_once()

    def test_on_deleted_any_file(self, handler, mock_watcher):
        """Test on_deleted event (processes all files)."""
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/deleted_file.py"

        with patch("asyncio.create_task") as mock_create_task:
            handler.on_deleted(event)

            # Should always process deletions (can't check patterns on deleted files)
            mock_create_task.assert_called_once()

    def test_on_moved_valid_destination(self, handler, mock_watcher):
        """Test on_moved event with valid destination."""
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/old_file.py"
        event.dest_path = "/test/new_file.py"

        with patch.object(handler, "_should_process", return_value=True):
            with patch("asyncio.create_task") as mock_create_task:
                handler.on_moved(event)

                # Should create MOVED change
                mock_create_task.assert_called_once()

    def test_on_moved_invalid_destination(self, handler, mock_watcher):
        """Test on_moved event with invalid destination (treated as deletion)."""
        event = MagicMock()
        event.is_directory = False
        event.src_path = "/test/file.py"
        event.dest_path = "/test/file.pyc"  # Will be filtered out

        with patch.object(handler, "_should_process", return_value=False):
            with patch("asyncio.create_task") as mock_create_task:
                handler.on_moved(event)

                # Should create DELETED change for old path
                mock_create_task.assert_called_once()


class TestFileWatcher:
    """Test FileWatcher class with real functionality."""

    @pytest.fixture
    def mock_indexer(self):
        """Create a mock DocumentIndexer."""
        indexer = MagicMock()
        indexer.config = MagicMock()
        indexer.config.document = MagicMock()
        indexer.config.document.file_patterns = ["*.py", "*.md", "*.txt"]

        indexer.scanner = MagicMock()
        indexer.scanner.generate_source_id = MagicMock(return_value="test_source_123")
        indexer.scanner.scan_folder = AsyncMock(
            return_value=[Path("/test/file1.py"), Path("/test/file2.md")]
        )

        indexer.index_folder = AsyncMock(return_value=MagicMock(file_count=5, total_chunks=20))
        indexer.index_file = AsyncMock(return_value=3)
        indexer.redis = MagicMock()

        return indexer

    @pytest.fixture
    def mock_graph_builder(self):
        """Create a mock KnowledgeGraphBuilder."""
        builder = MagicMock()
        builder.build_from_documents = AsyncMock()
        return builder

    @pytest.fixture
    def file_watcher(self, mock_indexer):
        """Create FileWatcher instance."""
        return FileWatcher(
            indexer=mock_indexer,
            debounce_seconds=0.1,
            batch_size=5,
            use_polling=True,  # Use polling to avoid watchdog dependency
        )

    @pytest.fixture
    def file_watcher_with_graph(self, mock_indexer, mock_graph_builder):
        """Create FileWatcher instance with graph builder."""
        return FileWatcher(
            indexer=mock_indexer,
            graph_builder=mock_graph_builder,
            debounce_seconds=0.1,
            batch_size=5,
            use_polling=True,
        )

    def test_watcher_initialization(self, file_watcher, mock_indexer):
        """Test FileWatcher initialization."""
        assert file_watcher.indexer == mock_indexer
        assert file_watcher.debounce_seconds == 0.1
        assert file_watcher.batch_size == 5
        assert file_watcher.use_polling is True
        assert file_watcher.is_running is False
        assert file_watcher.watched_sources == {}
        assert file_watcher.pending_changes == {}

    @pytest.mark.asyncio
    async def test_start_stop_polling_mode(self, file_watcher):
        """Test starting and stopping file watcher in polling mode."""
        assert not file_watcher.is_running

        # Start watcher
        await file_watcher.start()
        assert file_watcher.is_running is True

        # Stop watcher
        await file_watcher.stop()
        assert file_watcher.is_running is False

    @pytest.mark.asyncio
    async def test_start_already_running(self, file_watcher):
        """Test starting file watcher when already running."""
        file_watcher.is_running = True

        # Should handle gracefully
        await file_watcher.start()
        assert file_watcher.is_running is True

    @pytest.mark.asyncio
    async def test_stop_not_running(self, file_watcher):
        """Test stopping file watcher when not running."""
        assert not file_watcher.is_running

        # Should handle gracefully
        await file_watcher.stop()
        assert file_watcher.is_running is False

    @pytest.mark.asyncio
    async def test_watch_directory(self, file_watcher, mock_indexer):
        """Test watching a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)

            # Start watcher first
            await file_watcher.start()

            source_id = await file_watcher.watch(path, recursive=True, file_patterns=["*.py"])

            assert source_id == "test_source_123"
            assert source_id in file_watcher.watched_sources

            watched = file_watcher.watched_sources[source_id]
            # Handle macOS symlink resolution (/var -> /private/var)
            assert watched.path.resolve() == path.resolve()
            assert watched.recursive is True
            assert watched.file_patterns == ["*.py"]

            # Verify indexer was called
            mock_indexer.index_folder.assert_called_once()

            await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_watch_directory_with_graph_builder(
        self, file_watcher_with_graph, mock_graph_builder
    ):
        """Test watching directory with knowledge graph building."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)

            await file_watcher_with_graph.start()
            source_id = await file_watcher_with_graph.watch(path)

            # Verify graph builder was called
            mock_graph_builder.build_from_documents.assert_called_once_with(source_id)

            await file_watcher_with_graph.stop()

    @pytest.mark.asyncio
    async def test_watch_nonexistent_directory(self, file_watcher):
        """Test watching non-existent directory."""
        path = Path("/nonexistent/directory")

        with pytest.raises(ValueError, match="Path does not exist"):
            await file_watcher.watch(path)

    @pytest.mark.asyncio
    async def test_watch_file_not_directory(self, file_watcher):
        """Test watching a file instead of directory."""
        with tempfile.NamedTemporaryFile() as temp_file:
            path = Path(temp_file.name)

            with pytest.raises(ValueError, match="Path is not a directory"):
                await file_watcher.watch(path)

    @pytest.mark.asyncio
    async def test_watch_already_watching(self, file_watcher):
        """Test watching directory that's already being watched."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)

            await file_watcher.start()

            # Watch first time
            source_id1 = await file_watcher.watch(path)

            # Watch again - should return same source ID
            source_id2 = await file_watcher.watch(path)

            assert source_id1 == source_id2

            await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_unwatch(self, file_watcher):
        """Test unwatching a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir)

            await file_watcher.start()
            source_id = await file_watcher.watch(path)

            # Verify it's being watched
            assert source_id in file_watcher.watched_sources

            # Unwatch
            result = await file_watcher.unwatch(source_id)
            assert result is True
            assert source_id not in file_watcher.watched_sources

            await file_watcher.stop()

    @pytest.mark.asyncio
    async def test_unwatch_not_watched(self, file_watcher):
        """Test unwatching a source that's not being watched."""
        result = await file_watcher.unwatch("nonexistent_source")
        assert result is False

    @pytest.mark.asyncio
    async def test_handle_change_basic(self, file_watcher):
        """Test basic change handling."""
        change = FileChange(
            path=Path("/test/file.py"),
            change_type=ChangeType.CREATED,
            metadata={"source_id": "test_src"},
        )

        # Set up callback to capture change
        captured_changes = []
        file_watcher.add_change_callback(lambda c: captured_changes.append(c))

        await file_watcher._handle_change(change)

        # Verify change was processed
        assert len(captured_changes) == 1
        assert captured_changes[0].path == change.path
        assert file_watcher.stats["changes_detected"] == 1
        assert len(file_watcher.change_history) == 1

    @pytest.mark.asyncio
    async def test_process_file_change(self, file_watcher, mock_indexer):
        """Test processing file creation/modification."""
        change = FileChange(
            path=Path("/test/file.py"),
            change_type=ChangeType.MODIFIED,
            metadata={"source_id": "test_src"},
        )

        mock_indexer.index_file.return_value = 3  # 3 chunks

        await file_watcher._process_file_change(change)

        # Verify indexer was called
        mock_indexer.index_file.assert_called_once_with(change.path, source_id="test_src")
        assert file_watcher.stats["reindex_count"] == 1

    @pytest.mark.asyncio
    async def test_process_deletion(self, file_watcher, mock_indexer):
        """Test processing file deletion."""
        change = FileChange(path=Path("/test/file.py"), change_type=ChangeType.DELETED)

        # Mock Redis scan for cleanup - needs to be AsyncMock since it's awaited
        mock_indexer.redis.redis.scan = AsyncMock(
            side_effect=[(0, [b"chunk:abc123", b"chunk:def456"])]
        )
        mock_indexer.redis.redis.delete = AsyncMock()

        await file_watcher._process_deletion(change)

        # Verify deletion cleanup
        mock_indexer.redis.redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_move(self, file_watcher):
        """Test processing file move."""
        change = FileChange(
            path=Path("/test/new_file.py"),
            change_type=ChangeType.MOVED,
            old_path=Path("/test/old_file.py"),
            metadata={"source_id": "test_src"},
        )

        # Mock the processing methods
        file_watcher._process_deletion = AsyncMock()
        file_watcher._process_file_change = AsyncMock()

        await file_watcher._process_move(change)

        # Should delete old and index new
        file_watcher._process_deletion.assert_called_once()
        file_watcher._process_file_change.assert_called_once()

    def test_add_remove_change_callback(self, file_watcher):
        """Test adding and removing change callbacks."""
        callback_calls = []

        def test_callback(change):
            callback_calls.append(change)

        # Add callback
        file_watcher.add_change_callback(test_callback)
        assert test_callback in file_watcher.change_callbacks

        # Remove callback
        file_watcher.remove_change_callback(test_callback)
        assert test_callback not in file_watcher.change_callbacks

    def test_get_stats(self, file_watcher):
        """Test getting file watcher statistics."""
        # Set up some test stats
        file_watcher.stats["changes_detected"] = 10
        file_watcher.stats["changes_processed"] = 8
        file_watcher.stats["errors"] = 1
        file_watcher.is_running = True

        stats = file_watcher.get_stats()

        assert stats["changes_detected"] == 10
        assert stats["changes_processed"] == 8
        assert stats["errors"] == 1
        assert stats["is_running"] is True
        assert stats["mode"] == "polling"
        assert stats["watched_sources"] == 0  # No sources added yet

    def test_get_change_history(self, file_watcher):
        """Test getting change history."""
        # Add some test changes
        changes = [
            FileChange(Path(f"/test/file{i}.py"), ChangeType.CREATED, timestamp=1000 + i)
            for i in range(5)
        ]

        file_watcher.change_history = changes

        history = file_watcher.get_change_history(limit=3)

        assert len(history) == 3
        assert all("path" in change for change in history)
        assert all("type" in change for change in history)
        assert all("time" in change for change in history)

    @pytest.mark.asyncio
    async def test_force_rescan(self, file_watcher, mock_indexer):
        """Test force rescan of watched sources."""
        # Add a watched source
        watched_source = WatchedSource(path=Path("/test"), source_id="test_src", recursive=True)
        file_watcher.watched_sources["test_src"] = watched_source

        # Mock indexer response
        mock_result = MagicMock()
        mock_result.file_count = 10
        mock_result.total_chunks = 50
        mock_indexer.index_folder.return_value = mock_result

        result = await file_watcher.force_rescan()

        assert result["sources_scanned"] == 1
        assert result["files_indexed"] == 10
        assert result["chunks_created"] == 50

        # Verify indexer was called with force_reindex
        mock_indexer.index_folder.assert_called_once_with(
            Path("/test"), recursive=True, force_reindex=True
        )

    @pytest.mark.asyncio
    async def test_force_rescan_specific_source(self, file_watcher, mock_indexer):
        """Test force rescan of specific source."""
        # Add watched sources
        source1 = WatchedSource(path=Path("/test1"), source_id="src1")
        source2 = WatchedSource(path=Path("/test2"), source_id="src2")

        file_watcher.watched_sources = {"src1": source1, "src2": source2}

        mock_result = MagicMock()
        mock_result.file_count = 5
        mock_result.total_chunks = 25
        mock_indexer.index_folder.return_value = mock_result

        # Rescan only src1
        result = await file_watcher.force_rescan(source_id="src1")

        assert result["sources_scanned"] == 1
        assert result["files_indexed"] == 5

        # Should only call indexer once for src1
        mock_indexer.index_folder.assert_called_once()

    @pytest.mark.asyncio
    async def test_force_rescan_nonexistent_source(self, file_watcher):
        """Test force rescan of non-existent source."""
        result = await file_watcher.force_rescan(source_id="nonexistent")

        assert result["sources_scanned"] == 0
        assert result["files_indexed"] == 0
        assert result["chunks_created"] == 0
