"""
Simplified unit tests for document processor without external dependencies.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from eol.rag_context.config import DocumentConfig, ChunkingConfig

# Mock external dependencies
sys.modules['magic'] = MagicMock()
sys.modules['pypdf'] = MagicMock()
sys.modules['docx'] = MagicMock()

from eol.rag_context.document_processor import DocumentProcessor
from bs4 import BeautifulSoup


class TestDocumentProcessor:
    """Test document processor functionality."""
    
    def test_init(self):
        """Test processor initialization."""
        doc_config = DocumentConfig()
        chunk_config = ChunkingConfig()
        
        processor = DocumentProcessor(doc_config, chunk_config)
        
        assert processor.doc_config == doc_config
        assert processor.chunk_config == chunk_config
    
    def test_chunk_text_simple(self):
        """Test simple text chunking."""
        doc_config = DocumentConfig()
        chunk_config = ChunkingConfig()
        chunk_config.use_semantic_chunking = False
        chunk_config.max_chunk_size = 10
        chunk_config.chunk_overlap = 2
        
        processor = DocumentProcessor(doc_config, chunk_config)
        
        content = "This is a test document with multiple words that should be chunked properly."
        chunks = processor._chunk_text(content)
        
        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)
        assert all("type" in chunk for chunk in chunks)
    
    def test_chunk_markdown_by_headers(self):
        """Test markdown chunking by headers."""
        doc_config = DocumentConfig()
        chunk_config = ChunkingConfig()
        
        processor = DocumentProcessor(doc_config, chunk_config)
        
        content = """# Header 1
Content under header 1.

## Header 2
Content under header 2.

### Header 3
Content under header 3."""
        
        chunks = processor._chunk_markdown_by_headers(content)
        
        assert len(chunks) == 3
        assert chunks[0]["header"] == "Header 1"
        assert chunks[1]["header"] == "Header 2"
        assert chunks[2]["header"] == "Header 3"
    
    def test_detect_language(self):
        """Test programming language detection."""
        doc_config = DocumentConfig()
        chunk_config = ChunkingConfig()
        
        processor = DocumentProcessor(doc_config, chunk_config)
        
        assert processor._detect_language(".py") == "python"
        assert processor._detect_language(".js") == "javascript"
        assert processor._detect_language(".ts") == "typescript"
        assert processor._detect_language(".go") == "go"
        assert processor._detect_language(".unknown") == "unknown"
    
    def test_chunk_code_by_lines(self):
        """Test code chunking by lines."""
        doc_config = DocumentConfig()
        chunk_config = ChunkingConfig()
        chunk_config.code_max_lines = 5
        
        processor = DocumentProcessor(doc_config, chunk_config)
        
        content = "\n".join([f"line {i}" for i in range(20)])
        chunks = processor._chunk_code_by_lines(content, "python")
        
        assert len(chunks) > 0
        assert all(chunk["language"] == "python" for chunk in chunks)
        assert all(chunk["type"] == "lines" for chunk in chunks)
        assert chunks[0]["start_line"] == 1
    
    def test_extract_headers(self):
        """Test header extraction from HTML."""
        doc_config = DocumentConfig()
        chunk_config = ChunkingConfig()
        
        processor = DocumentProcessor(doc_config, chunk_config)
        
        html = """
        <h1>Main Title</h1>
        <h2>Subtitle</h2>
        <h3>Section</h3>
        """
        soup = BeautifulSoup(html, 'html.parser')
        headers = processor._extract_headers(soup)
        
        assert len(headers) == 3
        assert headers[0]["level"] == 1
        assert headers[0]["text"] == "Main Title"
        assert headers[1]["level"] == 2
        assert headers[2]["level"] == 3
    
    def test_chunk_structured_data(self):
        """Test structured data chunking."""
        doc_config = DocumentConfig()
        chunk_config = ChunkingConfig()
        
        processor = DocumentProcessor(doc_config, chunk_config)
        
        # Test dict chunking
        data = {"key1": "value1", "key2": "value2"}
        chunks = processor._chunk_structured_data(data, "json")
        
        assert len(chunks) == 2
        assert all(chunk["type"] == "object_field" for chunk in chunks)
        
        # Test list chunking
        data = ["item1", "item2", "item3"]
        chunks = processor._chunk_structured_data(data, "json")
        
        assert len(chunks) == 3
        assert all(chunk["type"] == "array_item" for chunk in chunks)
    
    def test_chunk_text_semantic(self):
        """Test semantic text chunking."""
        doc_config = DocumentConfig()
        chunk_config = ChunkingConfig()
        chunk_config.use_semantic_chunking = True
        chunk_config.max_chunk_size = 10
        
        processor = DocumentProcessor(doc_config, chunk_config)
        
        content = """First paragraph here.

Second paragraph here.

Third paragraph here."""
        
        chunks = processor._chunk_text(content)
        
        assert len(chunks) > 0
        assert all(chunk["type"] == "semantic" for chunk in chunks)