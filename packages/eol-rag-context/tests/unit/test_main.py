"""
Unit tests for main CLI module.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import after mocking
from eol.rag_context import main


class TestMainCLI:
    """Test main CLI functionality."""

    @patch("eol.rag_context.main.asyncio")
    @patch("eol.rag_context.main.EOLRAGContextServer")
    @patch("eol.rag_context.main.RAGConfig")
    def test_main_function_no_args(self, mock_config, mock_server, mock_asyncio):
        """Test main entry point with no arguments."""
        mock_config.return_value = MagicMock()
        mock_server_instance = MagicMock()
        mock_server_instance.run = MagicMock()
        mock_server.return_value = mock_server_instance
        mock_asyncio.run = MagicMock()

        # Mock sys.argv to have just the script name
        with patch("sys.argv", ["eol-rag-context"]):
            main.main()

            # Verify server was created with default config
            mock_config.assert_called_once()
            mock_server.assert_called_once()
            mock_asyncio.run.assert_called_once()

    @patch("eol.rag_context.main.asyncio")
    @patch("eol.rag_context.main.EOLRAGContextServer")
    @patch("eol.rag_context.main.RAGConfig")
    def test_main_function_with_config(self, mock_config, mock_server, mock_asyncio):
        """Test main entry point with config file argument."""
        mock_config.from_file.return_value = MagicMock()
        mock_server_instance = MagicMock()
        mock_server_instance.run = MagicMock()
        mock_server.return_value = mock_server_instance
        mock_asyncio.run = MagicMock()

        # Mock sys.argv with config file
        with patch("sys.argv", ["eol-rag-context", "config.json"]):
            main.main()

            # Verify server was created with file config
            mock_config.from_file.assert_called_once()
            mock_server.assert_called_once()
            mock_asyncio.run.assert_called_once()

    @patch("eol.rag_context.main.Path")
    @patch("eol.rag_context.main.EOLRAGContextServer")
    @patch("eol.rag_context.main.RAGConfig")
    def test_main_function_config_error(self, mock_config, mock_server, mock_path):
        """Test main entry point with config loading error."""
        mock_path.return_value = MagicMock()
        mock_config.from_file.side_effect = Exception("Config error")

        def exit_side_effect(code):
            if code == 1:
                raise SystemExit(code)  # Raise exception for error exit
            return None

        with patch("sys.argv", ["eol-rag-context", "bad_config.json"]):
            with patch("sys.exit", side_effect=exit_side_effect) as mock_exit:
                with patch("builtins.print"):
                    with pytest.raises(SystemExit) as exc_info:
                        main.main()
                    assert exc_info.value.code == 1

    @patch("eol.rag_context.main.asyncio")
    @patch("eol.rag_context.main.EOLRAGContextServer")
    @patch("eol.rag_context.main.RAGConfig")
    def test_main_function_server_error(self, mock_config, mock_server, mock_asyncio):
        """Test main entry point with server runtime error."""
        mock_config.return_value = MagicMock()
        mock_server_instance = MagicMock()
        mock_server.return_value = mock_server_instance
        mock_asyncio.run.side_effect = Exception("Server error")

        with patch("sys.argv", ["eol-rag-context"]):
            with patch("sys.exit") as mock_exit:
                main.main()
                mock_exit.assert_called_with(1)

    def test_help_output(self):
        """Test help message display."""

        def exit_side_effect(code):
            if code == 0:
                raise SystemExit(code)  # Raise exception for help exit
            return None  # Let other sys.exit calls be mocked normally

        with patch("sys.argv", ["eol-rag-context", "--help"]):
            with patch("sys.exit", side_effect=exit_side_effect) as mock_exit:
                with patch("builtins.print") as mock_print:
                    with pytest.raises(SystemExit) as exc_info:
                        main.main()
                    assert exc_info.value.code == 0
                    mock_print.assert_called()

    def test_help_output_short_flag(self):
        """Test help message display with -h flag."""

        def exit_side_effect(code):
            if code == 0:
                raise SystemExit(code)  # Raise exception for help exit
            return None  # Let other sys.exit calls be mocked normally

        with patch("sys.argv", ["eol-rag-context", "-h"]):
            with patch("sys.exit", side_effect=exit_side_effect) as mock_exit:
                with patch("builtins.print") as mock_print:
                    with pytest.raises(SystemExit) as exc_info:
                        main.main()
                    assert exc_info.value.code == 0
                    mock_print.assert_called()
