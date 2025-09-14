"""Integration tests for document indexing.

Tests real indexing workflow with Redis.

"""

import tempfile
from pathlib import Path

import pytest

from eol.rag_context import indexer


@pytest.mark.integration
class TestIndexingIntegration:
    """Test document indexing with real components."""

    @pytest.mark.asyncio
    async def test_index_single_file(self, indexer_instance, temp_test_directory):
        """Test indexing a single file."""
        test_file = temp_test_directory / "test.py"

        # Index the file
        result = await indexer_instance.index_file(test_file, "test_source_1")

        assert result is not None
        assert result.source_id == "test_source_1"
        assert result.chunks > 0
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_index_folder(self, server_instance, temp_test_directory):
        """Test indexing an entire folder via non-blocking API."""
        # Start non-blocking indexing
        result = await server_instance.index_directory(temp_test_directory, recursive=True)

        assert result is not None
        assert result["status"] == "started"
        assert "task_id" in result

        # Wait for indexing to complete
        task_id = result["task_id"]
        max_wait = 30  # 30 seconds max wait
        wait_time = 0

        while wait_time < max_wait:
            status_result = await server_instance.task_manager.get_task_status(task_id)
            if status_result and status_result.status.value in ["completed", "failed"]:
                break
            await asyncio.sleep(1)
            wait_time += 1

        # Verify indexing completed successfully
        final_status = await server_instance.task_manager.get_task_status(task_id)
        assert final_status is not None
        assert final_status.status.value == "completed"
        assert final_status.total_files > 0
        assert final_status.completed_files > 0

    @pytest.mark.asyncio
    async def test_hierarchical_indexing(self, indexer_instance, temp_test_directory):
        """Test hierarchical document indexing (concepts -> sections -> chunks)."""
        # Index a markdown file with clear structure
        md_file = temp_test_directory / "README.md"

        result = await indexer_instance.index_file(md_file, "hier_source")

        assert result is not None

        # The indexer should create hierarchical structure
        # Level 1: Concepts (main topics)
        # Level 2: Sections (subtopics)
        # Level 3: Chunks (actual content)
        assert result.chunks > 0

    @pytest.mark.asyncio
    async def test_folder_scanner(self, indexer_instance):
        """Test folder scanning functionality."""
        scanner = indexer.FolderScanner(indexer_instance.config)

        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Create test structure
            (test_dir / "file1.py").write_text("print('test')")
            (test_dir / "file2.md").write_text("# Test")
            (test_dir / ".git").mkdir()
            (test_dir / ".git" / "config").write_text("git config")
            (test_dir / "__pycache__").mkdir()
            (test_dir / "__pycache__" / "test.pyc").write_text("compiled")

            # Scan folder
            files = await scanner.scan_folder(test_dir)

            # Should exclude .git and __pycache__
            assert len(files) == 2
            file_paths = [f.name for f in files]
            assert "file1.py" in file_paths
            assert "file2.md" in file_paths
            assert "test.pyc" not in file_paths
            assert "config" not in file_paths

    @pytest.mark.asyncio
    async def test_source_management(self, server_instance, indexer_instance, temp_test_directory):
        """Test source management operations."""
        # Index with a specific source ID using non-blocking API
        source_id = "test_source_mgmt"

        # Start non-blocking indexing
        result = await server_instance.index_directory(temp_test_directory, recursive=True)
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

        # Get the actual source_id from the completed task
        final_status = await server_instance.task_manager.get_task_status(task_id)
        actual_source_id = final_status.source_id

        # List sources
        sources = await indexer_instance.list_sources()
        assert len(sources) > 0

        # Find our source
        our_source = None
        for source in sources:
            if source.source_id == actual_source_id:
                our_source = source
                break

        assert our_source is not None
        assert our_source.file_count > 0

        # Remove source
        removed = await indexer_instance.remove_source(actual_source_id)
        assert removed

        # Verify removal
        sources_after = await indexer_instance.list_sources()
        source_ids_after = [s.source_id for s in sources_after]
        assert actual_source_id not in source_ids_after

    @pytest.mark.asyncio
    async def test_incremental_indexing(self, server_instance):
        """Test incremental indexing with file changes."""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Initial file
            file1 = test_dir / "file1.txt"
            file1.write_text("Initial content")

            # First index
            result1 = await server_instance.index_directory(test_dir, recursive=True)
            task_id1 = result1["task_id"]

            # Wait for first indexing to complete
            max_wait = 30
            wait_time = 0
            while wait_time < max_wait:
                status = await server_instance.task_manager.get_task_status(task_id1)
                if status and status.status.value in ["completed", "failed"]:
                    break
                await asyncio.sleep(1)
                wait_time += 1

            first_status = await server_instance.task_manager.get_task_status(task_id1)
            initial_chunks = first_status.total_chunks if first_status else 0

            # Add another file
            file2 = test_dir / "file2.txt"
            file2.write_text("Additional content")

            # Re-index
            result2 = await server_instance.index_directory(test_dir, recursive=True)
            task_id2 = result2["task_id"]

            # Wait for second indexing to complete
            wait_time = 0
            while wait_time < max_wait:
                status = await server_instance.task_manager.get_task_status(task_id2)
                if status and status.status.value in ["completed", "failed"]:
                    break
                await asyncio.sleep(1)
                wait_time += 1

            second_status = await server_instance.task_manager.get_task_status(task_id2)

            # Should have more chunks
            assert second_status is not None
            assert second_status.total_chunks > initial_chunks
            assert second_status.total_files == 2

    @pytest.mark.asyncio
    async def test_error_recovery(self, indexer_instance):
        """Test error handling during indexing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Create a mix of valid and problematic files
            (test_dir / "good.txt").write_text("Valid content")

            # Create a file with unusual permissions (if possible)
            bad_file = test_dir / "bad.txt"
            bad_file.write_text("Content")

            # Try to make it unreadable (platform-dependent)
            try:
                bad_file.chmod(0o000)
            except OSError:
                pass  # May not work on all platforms

            # Index should handle errors gracefully
            result = await indexer_instance.index_folder(test_dir)

            # Should index at least the good file
            assert result.file_count >= 1

            # Clean up permissions
            try:
                bad_file.chmod(0o644)
            except OSError:
                pass

    @pytest.mark.asyncio
    async def test_metadata_extraction(self, indexer_instance, temp_test_directory):
        """Test metadata extraction during indexing."""
        test_file = temp_test_directory / "test.py"

        # Index with metadata tracking
        result = await indexer_instance.index_file(test_file, "metadata_test")

        assert result is not None

        # Metadata should be stored with documents
        # This would be verified through Redis queries in real scenario

    @pytest.mark.asyncio
    async def test_concurrent_indexing(self, server_instance, temp_test_directory):
        """Test concurrent directory indexing."""
        import asyncio
        import tempfile

        # Create multiple temporary directories for concurrent indexing
        temp_dirs = []
        for i in range(3):
            tmpdir = tempfile.mkdtemp()
            temp_dir = Path(tmpdir)
            (temp_dir / f"file_{i}.py").write_text(f"# File {i}\nprint('concurrent test {i}')")
            temp_dirs.append(temp_dir)

        try:
            # Index directories concurrently
            tasks = [
                server_instance.index_directory(temp_dir, recursive=True) for temp_dir in temp_dirs
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify all started successfully
            successful = [r for r in results if not isinstance(r, Exception)]
            assert len(successful) == len(temp_dirs)

            # Wait for all tasks to complete
            task_ids = [r["task_id"] for r in successful]
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
                assert final_status.completed_files > 0

        finally:
            # Clean up temporary directories
            import shutil

            for temp_dir in temp_dirs:
                shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_indexing_stats(self, indexer_instance, temp_test_directory):
        """Test indexing statistics."""
        # Index some files
        await indexer_instance.index_folder(temp_test_directory)

        # Get stats
        stats = indexer_instance.get_stats()

        assert "total_documents" in stats
        assert "total_chunks" in stats
        assert "sources" in stats
        assert stats["total_documents"] > 0
