"""
Unit tests for document processor.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

from eol.rag_context.config import ChunkingConfig, DocumentConfig
from eol.rag_context.document_processor import DocumentProcessor, ProcessedDocument


class TestDocumentProcessor:
    """Test document processor functionality."""

    @pytest.fixture
    def processor(self):
        """Create document processor."""
        doc_config = DocumentConfig()
        chunk_config = ChunkingConfig()
        return DocumentProcessor(doc_config, chunk_config)

    @pytest.mark.asyncio
    async def test_process_markdown(self, processor, sample_documents):
        """Test processing markdown files."""
        md_file = sample_documents["markdown"]

        doc = await processor.process_file(md_file)

        assert doc is not None
        assert doc.doc_type == "markdown"
        assert doc.file_path == md_file
        assert len(doc.chunks) > 0
        assert "headers" in doc.metadata

        # Check header extraction
        headers = doc.metadata["headers"]
        assert len(headers) > 0
        assert any(h["text"] == "Test Document" for h in headers)

    @pytest.mark.asyncio
    async def test_process_python(self, processor, sample_documents):
        """Test processing Python files."""
        py_file = sample_documents["python"]

        doc = await processor.process_file(py_file)

        assert doc is not None
        assert doc.doc_type == "code"
        assert doc.language == "python"
        assert len(doc.chunks) > 0

        # Check that functions are detected
        chunk_contents = [c["content"] for c in doc.chunks]
        assert any("factorial" in content for content in chunk_contents)
        assert any("TestClass" in content for content in chunk_contents)

    @pytest.mark.asyncio
    async def test_process_json(self, processor, sample_documents):
        """Test processing JSON files."""
        json_file = sample_documents["json"]

        doc = await processor.process_file(json_file)

        assert doc is not None
        assert doc.doc_type == "structured"  # JSON files are processed as structured
        assert len(doc.chunks) > 0
        assert "keys" in doc.metadata
        assert "name" in doc.metadata["keys"]

    @pytest.mark.asyncio
    async def test_process_text(self, processor, sample_documents):
        """Test processing plain text files."""
        txt_file = sample_documents["text"]

        doc = await processor.process_file(txt_file)

        assert doc is not None
        assert doc.doc_type == "text"
        assert len(doc.chunks) > 0
        assert doc.metadata["lines"] > 0

    @pytest.mark.asyncio
    async def test_chunk_by_headers(self, processor):
        """Test markdown chunking by headers."""
        content = """# Header 1
Content under header 1.

## Header 2
Content under header 2.

### Header 3
Content under header 3.

## Another Header 2
More content here."""

        chunks = processor._chunk_markdown_by_headers(content)

        assert len(chunks) == 4
        assert all("header" in chunk["metadata"] for chunk in chunks)
        assert chunks[0]["metadata"]["header"] == "Header 1"

    def test_chunk_text_semantic(self, processor):
        """Test semantic text chunking."""
        processor.chunk_config.use_semantic_chunking = True
        processor.chunk_config.max_chunk_size = 10  # Small for testing

        content = """First paragraph with some content.

Second paragraph with different content.

Third paragraph with more information."""

        chunks = processor._chunk_text(content)

        assert len(chunks) > 0
        assert all(chunk["type"] == "semantic" for chunk in chunks)
        assert all("content" in chunk for chunk in chunks)

    def test_chunk_code_by_lines(self, processor):
        """Test code chunking by lines."""
        processor.chunk_config.code_max_lines = 5

        content = "\n".join([f"line {i}" for i in range(20)])

        chunks = processor._chunk_code_by_lines(content, "python")

        assert len(chunks) > 0
        assert all(chunk["metadata"]["language"] == "python" for chunk in chunks)
        assert all(chunk["type"] == "lines" for chunk in chunks)
        assert chunks[0]["metadata"]["start_line"] == 1

    @pytest.mark.asyncio
    async def test_process_nonexistent_file(self, processor):
        """Test processing non-existent file."""
        doc = await processor.process_file(Path("/nonexistent/file.txt"))
        assert doc is None

    @pytest.mark.asyncio
    async def test_process_large_file(self, processor, temp_dir):
        """Test handling of large files."""
        # Create a file larger than max size
        large_file = temp_dir / "large.txt"
        processor.doc_config.max_file_size_mb = 0.001  # 1KB limit

        # Write 10KB of data
        large_file.write_text("x" * 10000)

        doc = await processor.process_file(large_file)
        assert doc is None  # Should skip large files


# Removed TestDocumentProcessorAdditional class - tests were for non-existent methods
