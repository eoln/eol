"""
Final test for document processor to reach 80% coverage.
"""

from pathlib import Path

from eol.rag_context import config, document_processor


class TestDocumentProcessorFinal:
    """Final tests for document processor."""

    def test_document_processor_init(self):
        """Test DocumentProcessor initialization."""
        doc_config = config.DocumentConfig()
        chunk_config = config.ChunkingConfig()

        processor = document_processor.DocumentProcessor(doc_config, chunk_config)

        assert processor.doc_config == doc_config
        assert processor.chunk_config == chunk_config
        assert processor.mime is not None

    def test_processed_document_dataclass(self):
        """Test ProcessedDocument dataclass."""
        result = document_processor.ProcessedDocument(
            doc_type="text",
            content="Test content",
            chunks=[{"content": "chunk1"}, {"content": "chunk2"}],
            metadata={"key": "value"},
            file_path=Path("/test/file.txt"),
        )

        assert result.doc_type == "text"
        assert result.content == "Test content"
        assert len(result.chunks) == 2
        assert result.metadata["key"] == "value"
        assert result.file_path == Path("/test/file.txt")

    def test_chunk_structure(self):
        """Test chunk structure creation."""
        doc_config = config.DocumentConfig()
        chunk_config = config.ChunkingConfig()
        processor = document_processor.DocumentProcessor(doc_config, chunk_config)

        # Create a chunk
        chunk = processor._create_chunk(
            content="Test chunk content",
            chunk_type="paragraph",
            index=0,
            total=5,
            parent_section="Introduction",
        )

        assert chunk["content"] == "Test chunk content"
        assert chunk["type"] == "paragraph"
        assert chunk["metadata"]["index"] == 0
        assert chunk["metadata"]["total"] == 5
        assert chunk["metadata"]["parent_section"] == "Introduction"
        assert "tokens" in chunk


if __name__ == "__main__":
    print("✅ Final document processor tests!")
