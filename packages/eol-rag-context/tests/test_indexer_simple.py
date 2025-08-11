"""
Simplified unit tests for indexer without Redis dependency.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import hashlib
import tempfile
import shutil

# Mock external dependencies
sys.modules["magic"] = MagicMock()
sys.modules["pypdf"] = MagicMock()
sys.modules["docx"] = MagicMock()
sys.modules["redis"] = MagicMock()
sys.modules["redis.asyncio"] = MagicMock()
sys.modules["redis.commands"] = MagicMock()
sys.modules["redis.commands.search"] = MagicMock()
sys.modules["redis.commands.search.field"] = MagicMock()
sys.modules["redis.commands.search.indexDefinition"] = MagicMock()
sys.modules["redis.commands.search.query"] = MagicMock()
sys.modules["watchdog"] = MagicMock()
sys.modules["watchdog.observers"] = MagicMock()
sys.modules["watchdog.events"] = MagicMock()
sys.modules["networkx"] = MagicMock()

from eol.rag_context.indexer import FolderScanner, DocumentMetadata
from eol.rag_context.config import RAGConfig


class TestFolderScanner:
    """Test folder scanner functionality."""

    @pytest.fixture
    def scanner(self):
        """Create folder scanner."""
        config = RAGConfig(data_dir=Path("/tmp/data"), index_dir=Path("/tmp/index"))
        return FolderScanner(config)

    def test_default_ignore_patterns(self, scanner):
        """Test default ignore patterns."""
        patterns = scanner._default_ignore_patterns()

        assert "**/.git/**" in patterns
        assert "**/node_modules/**" in patterns
        assert "**/__pycache__/**" in patterns
        assert "**/.DS_Store" in patterns

    def test_should_ignore_default_patterns(self, scanner):
        """Test file ignore logic with default patterns."""
        # Create mock paths
        git_file = Path("/project/.git/config")
        node_file = Path("/project/node_modules/package.json")
        normal_file = Path("/project/src/main.py")

        assert scanner._should_ignore(git_file)
        assert scanner._should_ignore(node_file)
        assert not scanner._should_ignore(normal_file)

    def test_should_ignore_file_size(self, scanner, tmp_path):
        """Test ignoring files by size."""
        # Create a small file
        small_file = tmp_path / "small.txt"
        small_file.write_text("small content")

        # Create a large file
        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * (101 * 1024 * 1024))  # 101MB

        assert not scanner._should_ignore(small_file)
        assert scanner._should_ignore(large_file)

    def test_generate_source_id(self, scanner):
        """Test source ID generation."""
        path = Path("/test/path")
        source_id = scanner.generate_source_id(path)

        assert isinstance(source_id, str)
        assert len(source_id) == 16

        # Should be deterministic
        source_id2 = scanner.generate_source_id(path)
        assert source_id == source_id2

        # Different paths should give different IDs
        different_path = Path("/different/path")
        different_id = scanner.generate_source_id(different_path)
        assert source_id != different_id

    def test_get_git_metadata_no_git(self, scanner, tmp_path):
        """Test git metadata extraction when not in git repo."""
        metadata = scanner._get_git_metadata(tmp_path)
        assert metadata == {}

    @pytest.mark.asyncio
    async def test_scan_folder_basic(self, scanner, tmp_path):
        """Test basic folder scanning."""
        # Create test files
        (tmp_path / "test.py").write_text("print('test')")
        (tmp_path / "test.md").write_text("# Test")
        (tmp_path / "test.txt").write_text("test")

        files = await scanner.scan_folder(tmp_path, recursive=False)

        assert len(files) > 0
        assert all(f.exists() for f in files)

    @pytest.mark.asyncio
    async def test_scan_folder_with_patterns(self, scanner, tmp_path):
        """Test folder scanning with file patterns."""
        # Create test files
        (tmp_path / "test.py").write_text("print('test')")
        (tmp_path / "test.md").write_text("# Test")
        (tmp_path / "test.txt").write_text("test")

        # Only Python files
        files = await scanner.scan_folder(tmp_path, recursive=False, file_patterns=["*.py"])

        assert len(files) == 1
        assert files[0].suffix == ".py"

    @pytest.mark.asyncio
    async def test_scan_folder_recursive(self, scanner, tmp_path):
        """Test recursive folder scanning."""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.py").write_text("root")
        (subdir / "sub.py").write_text("sub")

        # Non-recursive
        files = await scanner.scan_folder(tmp_path, recursive=False, file_patterns=["*.py"])
        assert len(files) == 1

        # Recursive
        files = await scanner.scan_folder(tmp_path, recursive=True, file_patterns=["*.py"])
        assert len(files) == 2

    @pytest.mark.asyncio
    async def test_scan_nonexistent_folder(self, scanner):
        """Test scanning non-existent folder."""
        with pytest.raises(ValueError, match="does not exist"):
            await scanner.scan_folder(Path("/nonexistent"))

    @pytest.mark.asyncio
    async def test_scan_file_not_directory(self, scanner, tmp_path):
        """Test scanning a file instead of directory."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        with pytest.raises(ValueError, match="not a directory"):
            await scanner.scan_folder(file_path)


class TestDocumentMetadata:
    """Test document metadata dataclass."""

    def test_metadata_creation(self):
        """Test creating document metadata."""
        metadata = DocumentMetadata(
            source_path="/test/file.py",
            source_id="test123",
            relative_path="file.py",
            file_type="code",
            file_size=1024,
            file_hash="abcd1234",
            modified_time=1234567890.0,
            indexed_at=1234567891.0,
            chunk_index=0,
            total_chunks=5,
            hierarchy_level=3,
        )

        assert metadata.source_path == "/test/file.py"
        assert metadata.source_id == "test123"
        assert metadata.file_type == "code"
        assert metadata.hierarchy_level == 3

    def test_metadata_optional_fields(self):
        """Test optional metadata fields."""
        metadata = DocumentMetadata(
            source_path="/test/file.py",
            source_id="test123",
            relative_path="file.py",
            file_type="code",
            file_size=1024,
            file_hash="abcd1234",
            modified_time=1234567890.0,
            indexed_at=1234567891.0,
            chunk_index=0,
            total_chunks=5,
            hierarchy_level=3,
            language="python",
            line_start=10,
            line_end=20,
            git_commit="abc123",
            git_branch="main",
        )

        assert metadata.language == "python"
        assert metadata.line_start == 10
        assert metadata.line_end == 20
        assert metadata.git_commit == "abc123"
        assert metadata.git_branch == "main"
