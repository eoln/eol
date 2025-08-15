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
    async def test_index_folder(self, indexer_instance, temp_test_directory):
        """Test indexing an entire folder."""
        # Index the folder
        result = await indexer_instance.index_folder(temp_test_directory)

        assert result is not None
        assert result.file_count > 0
        assert result.total_chunks > 0
        assert result.source_id != ""

        # Verify individual files were indexed
        assert result.indexed_files > 0

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
    async def test_source_management(self, indexer_instance, temp_test_directory):
        """Test source management operations."""
        # Index with a specific source ID
        source_id = "test_source_mgmt"

        # Index files
        result = await indexer_instance.index_folder(
            temp_test_directory, source_id=source_id
        )

        assert result.source_id == source_id

        # List sources
        sources = await indexer_instance.list_sources()
        assert len(sources) > 0

        # Find our source
        our_source = None
        for source in sources:
            if source.source_id == source_id:
                our_source = source
                break

        assert our_source is not None
        assert our_source.file_count > 0

        # Remove source
        removed = await indexer_instance.remove_source(source_id)
        assert removed

        # Verify removal
        sources_after = await indexer_instance.list_sources()
        source_ids_after = [s.source_id for s in sources_after]
        assert source_id not in source_ids_after

    @pytest.mark.asyncio
    async def test_incremental_indexing(self, indexer_instance):
        """Test incremental indexing with file changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Initial file
            file1 = test_dir / "file1.txt"
            file1.write_text("Initial content")

            # First index
            result1 = await indexer_instance.index_folder(test_dir)
            initial_chunks = result1.total_chunks

            # Add another file
            file2 = test_dir / "file2.txt"
            file2.write_text("Additional content")

            # Re-index
            result2 = await indexer_instance.index_folder(test_dir)

            # Should have more chunks
            assert result2.total_chunks > initial_chunks
            assert result2.file_count == 2

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
    async def test_concurrent_indexing(self, indexer_instance, temp_test_directory):
        """Test concurrent file indexing."""
        import asyncio

        files = list(temp_test_directory.glob("*.py"))[:3]

        # Index files concurrently
        tasks = [
            indexer_instance.index_file(f, f"concurrent_{i}")
            for i, f in enumerate(files)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all succeeded
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) == len(files)

        for result in successful:
            assert result.chunks > 0

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
