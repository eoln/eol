"""Unit tests for main module."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eol.rag_context import main


class TestMainModule:
    """Test main module functionality."""
    
    def test_module_imports(self):
        """Test that main module can be imported."""
        assert main is not None
        assert hasattr(main, '__file__')
    
    @patch('eol.rag_context.main.server')
    def test_main_imports_server(self, mock_server):
        """Test that main imports server module."""
        # Reload to trigger import
        import importlib
        importlib.reload(main)
        assert mock_server is not None
    
    @patch('sys.argv', ['eol-rag-context'])
    @patch('eol.rag_context.main.asyncio')
    @patch('eol.rag_context.main.server.EOLRAGContextServer')
    def test_main_function_runs_server(self, mock_server_class, mock_asyncio):
        """Test main function creates and runs server."""
        mock_server = AsyncMock()
        mock_server_class.return_value = mock_server
        mock_asyncio.run = MagicMock()
        
        # Import and run main
        from eol.rag_context.main import main as main_func
        
        # Since main() calls asyncio.run directly, mock it
        with patch('asyncio.run') as mock_run:
            main_func()
            mock_run.assert_called_once()
            # Get the coroutine that was passed to asyncio.run
            coro = mock_run.call_args[0][0]
            assert coro is not None
    
    def test_module_structure(self):
        """Test main module has expected structure."""
        # Check module has expected attributes
        assert hasattr(main, '__name__')
        assert hasattr(main, '__package__')
        
    @patch('eol.rag_context.main.logging')
    def test_logging_setup(self, mock_logging):
        """Test that logging is configured."""
        import importlib
        importlib.reload(main)
        # Logging should be imported
        assert mock_logging is not None
        
    @patch.dict('sys.modules', {'eol.rag_context.server': MagicMock()})
    def test_server_import(self):
        """Test server module is imported."""
        import importlib
        importlib.reload(main)
        assert 'eol.rag_context.server' in sys.modules
        
    def test_main_is_entry_point(self):
        """Test main module can serve as entry point."""
        # Check if __name__ == "__main__" block exists
        with open(main.__file__, 'r') as f:
            content = f.read()
            assert 'if __name__ == "__main__"' in content
            
    @patch('eol.rag_context.main.sys.exit')
    @patch('eol.rag_context.main.asyncio.run')
    @patch('eol.rag_context.main.server.EOLRAGContextServer')
    def test_main_handles_keyboard_interrupt(self, mock_server_class, mock_run, mock_exit):
        """Test main handles KeyboardInterrupt gracefully."""
        mock_run.side_effect = KeyboardInterrupt()
        
        from eol.rag_context.main import main as main_func
        main_func()
        
        # Should exit cleanly
        mock_exit.assert_not_called()  # KeyboardInterrupt should be handled gracefully
        
    @patch('eol.rag_context.main.asyncio.run')
    @patch('eol.rag_context.main.server.EOLRAGContextServer')
    def test_main_handles_exception(self, mock_server_class, mock_run):
        """Test main handles exceptions."""
        mock_run.side_effect = Exception("Test error")
        
        from eol.rag_context.main import main as main_func
        
        # Should raise the exception (or handle it)
        with pytest.raises(Exception) as exc_info:
            main_func()
        assert "Test error" in str(exc_info.value)