"""
Final push to achieve 80% test coverage.
"""

from pathlib import Path
from unittest.mock import MagicMock

from eol.rag_context import config, indexer


class TestIndexerMissingCoverage:
    """Test missing branches in indexer to reach 80%."""

    def test_folder_scanner_creation(self):
        """Test FolderScanner creation with custom patterns."""
        rag_config = config.RAGConfig()
        # Add custom file patterns
        rag_config.document.file_patterns.append("*.custom")

        scanner = indexer.FolderScanner(rag_config)

        assert scanner is not None
        # FolderScanner stores patterns in config
        assert "*.custom" in scanner.config.document.file_patterns
        assert scanner.config == rag_config

    def test_indexed_source_dataclass(self):
        """Test IndexedSource dataclass."""
        source = indexer.IndexedSource(
            source_id="test_123",
            path=Path("/test/path"),
            indexed_at=1234567890.0,
            file_count=5,
            total_chunks=20,
            indexed_files=5,
            metadata={"key": "value"},
        )

        assert source.source_id == "test_123"
        assert source.file_count == 5
        assert source.total_chunks == 20
        assert source.metadata["key"] == "value"

    def test_index_result_dataclass(self):
        """Test IndexResult dataclass."""
        result = indexer.IndexResult(
            source_id="test_123", chunks=50, files=10, errors=["error1", "error2"]
        )

        assert result.files == 10
        assert result.chunks == 50
        assert len(result.errors) == 2
        assert result.source_id == "test_123"

    def test_document_metadata_dataclass(self):
        """Test DocumentMetadata dataclass."""
        metadata = indexer.DocumentMetadata(
            source_path=Path("/test/file.txt"),
            source_id="src_123",
            relative_path="file.txt",
            file_type="text",
            file_size=1024,
            file_hash="abc123",
            modified_time=1234567890.0,
            indexed_at=1234567891.0,
            chunk_index=5,
            total_chunks=10,
            hierarchy_level=2,
        )

        assert metadata.source_id == "src_123"
        assert metadata.file_type == "text"
        assert metadata.file_size == 1024
        assert metadata.file_hash == "abc123"
        assert metadata.chunk_index == 5
        assert metadata.total_chunks == 10
        assert metadata.hierarchy_level == 2

    def test_document_indexer_scanner_property(self):
        """Test DocumentIndexer scanner property."""
        mock_config = config.RAGConfig()
        mock_processor = MagicMock()
        mock_embedding = MagicMock()
        mock_redis = MagicMock()

        idx = indexer.DocumentIndexer(mock_config, mock_processor, mock_embedding, mock_redis)

        # Access the scanner property
        scanner = idx.scanner
        assert isinstance(scanner, indexer.FolderScanner)
        assert scanner.config == mock_config


class TestConfigExtraProperties:
    """Test additional config properties."""

    def test_document_config_properties(self):
        """Test DocumentConfig properties."""
        doc_config = config.DocumentConfig()

        # Test default values
        assert doc_config.extract_metadata is True
        assert doc_config.detect_language is True
        assert doc_config.parse_code_structure is True
        assert doc_config.max_file_size_mb == 100
        assert doc_config.skip_binary_files is True

        # Test file patterns
        assert "*.py" in doc_config.file_patterns
        assert "*.md" in doc_config.file_patterns
        assert "*.json" in doc_config.file_patterns

    def test_index_config_properties(self):
        """Test IndexConfig properties."""
        idx_config = config.IndexConfig()

        # Test HNSW parameters
        assert idx_config.algorithm == "HNSW"
        assert idx_config.distance_metric == "COSINE"
        assert idx_config.m == 16
        assert idx_config.ef_construction == 200
        assert idx_config.ef_runtime == 10

        # Test hierarchy settings
        assert idx_config.hierarchy_levels == 3
        assert idx_config.concept_prefix == "concept:"
        assert idx_config.section_prefix == "section:"
        assert idx_config.chunk_prefix == "chunk:"

    def test_context_config_properties(self):
        """Test ContextConfig properties."""
        ctx_config = config.ContextConfig()

        # Test context settings
        assert ctx_config.use_hierarchical_retrieval is True
        assert ctx_config.progressive_loading is True
        assert ctx_config.remove_redundancy is True
        assert ctx_config.redundancy_threshold == 0.9

        # Test default values
        assert ctx_config.max_context_tokens == 32000
        assert ctx_config.reserve_tokens_for_response == 4000
        assert ctx_config.default_top_k == 10
        assert ctx_config.min_relevance_score == 0.7


if __name__ == "__main__":
    print("✅ Final coverage tests to reach 80%!")
