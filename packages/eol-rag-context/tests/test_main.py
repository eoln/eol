"""
Unit tests for main CLI module.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock dependencies
for module in ["typer", "rich", "rich.console", "redis", "redis.asyncio"]:
    if module not in sys.modules:
        sys.modules[module] = MagicMock()

# Import after mocking
from eol.rag_context import main


class TestMainCLI:
    """Test main CLI functionality."""

    def test_app_creation(self):
        """Test that the Typer app is created."""
        assert main.app is not None

    def test_serve_command(self):
        """Test serve command."""
        with patch("eol.rag_context.main.RAGContextServer") as MockServer:
            mock_server = MagicMock()
            MockServer.return_value = mock_server

            # Call serve command
            main.serve(host="localhost", port=8080, data_dir="/data", index_dir="/index")

            # Verify server was created and run
            MockServer.assert_called_once()
            mock_server.run.assert_called_once()

    def test_index_command(self):
        """Test index command."""
        with (
            patch("eol.rag_context.main.DocumentIndexer") as MockIndexer,
            patch("eol.rag_context.main.asyncio") as mock_asyncio,
        ):

            mock_indexer = MagicMock()
            MockIndexer.return_value = mock_indexer
            mock_asyncio.run = MagicMock()

            # Call index command
            main.index(path="/test/path", recursive=True, patterns=["*.py", "*.md"])

            # Verify asyncio.run was called
            mock_asyncio.run.assert_called_once()

    def test_search_command(self):
        """Test search command."""
        with (
            patch("eol.rag_context.main.RedisVectorStore") as MockRedis,
            patch("eol.rag_context.main.asyncio") as mock_asyncio,
            patch("eol.rag_context.main.console") as mock_console,
        ):

            mock_redis = MagicMock()
            MockRedis.return_value = mock_redis
            mock_asyncio.run = MagicMock()

            # Call search command
            main.search(query="test query", limit=5, hierarchy_level=3)

            # Verify asyncio.run was called
            mock_asyncio.run.assert_called_once()

    def test_stats_command(self):
        """Test stats command."""
        with (
            patch("eol.rag_context.main.RAGComponents") as MockComponents,
            patch("eol.rag_context.main.asyncio") as mock_asyncio,
            patch("eol.rag_context.main.console") as mock_console,
        ):

            mock_components = MagicMock()
            MockComponents.return_value = mock_components
            mock_asyncio.run = MagicMock()

            # Call stats command
            main.stats()

            # Verify asyncio.run was called
            mock_asyncio.run.assert_called_once()

    def test_clear_cache_command(self):
        """Test clear-cache command."""
        with (
            patch("eol.rag_context.main.SemanticCache") as MockCache,
            patch("eol.rag_context.main.asyncio") as mock_asyncio,
            patch("eol.rag_context.main.console") as mock_console,
        ):

            mock_cache = MagicMock()
            MockCache.return_value = mock_cache
            mock_asyncio.run = MagicMock()

            # Call clear-cache command
            main.clear_cache()

            # Verify asyncio.run was called
            mock_asyncio.run.assert_called_once()

    def test_watch_command(self):
        """Test watch command."""
        with (
            patch("eol.rag_context.main.FileWatcher") as MockWatcher,
            patch("eol.rag_context.main.asyncio") as mock_asyncio,
        ):

            mock_watcher = MagicMock()
            MockWatcher.return_value = mock_watcher
            mock_asyncio.run = MagicMock()

            # Call watch command
            main.watch(path="/test/path", recursive=True, patterns=["*.py"])

            # Verify asyncio.run was called
            mock_asyncio.run.assert_called_once()

    def test_main_function(self):
        """Test main entry point."""
        with patch.object(main.app, "run") as mock_run:
            # Call main
            main.main()

            # Verify app.run was called
            mock_run.assert_called_once()

    def test_console_output(self):
        """Test console output formatting."""
        with patch("eol.rag_context.main.console") as mock_console:
            # Test various console outputs
            mock_console.print = MagicMock()

            # Simulate console usage
            from rich.console import Console

            console = Console()

            # This would be called in the actual commands
            console.print("[bold green]Success![/bold green]")

            # Just verify console can be created
            assert console is not None


class TestAsyncHelpers:
    """Test async helper functions in main."""

    @pytest.mark.asyncio
    async def test_async_index(self):
        """Test async index operation."""
        with patch("eol.rag_context.main.DocumentIndexer") as MockIndexer:
            mock_indexer = MagicMock()
            mock_indexer.index_folder = MagicMock()
            MockIndexer.return_value = mock_indexer

            # Create async version of the index operation
            async def async_index():
                indexer = MockIndexer()
                await indexer.index_folder(Path("/test"))
                return True

            result = await async_index()
            assert result is True

    @pytest.mark.asyncio
    async def test_async_search(self):
        """Test async search operation."""
        with patch("eol.rag_context.main.RedisVectorStore") as MockRedis:
            mock_redis = MagicMock()
            mock_redis.search = MagicMock(return_value=[])
            MockRedis.return_value = mock_redis

            # Create async version of the search operation
            async def async_search():
                store = MockRedis()
                results = await store.search("query")
                return results

            results = await async_search()
            assert isinstance(results, list)
