"""
Final test to push coverage over 80%.
"""

# Removed unused imports

from eol.rag_context import config, document_processor


class TestFinalCoverage:
    """Final tests to reach 80%."""

    def test_chunking_config_code_settings(self):
        """Test code-specific chunking settings."""
        chunk_config = config.ChunkingConfig()

        # Test code chunking settings
        assert chunk_config.code_chunk_by_function is True
        assert chunk_config.code_max_lines == 100
        assert chunk_config.respect_document_structure is True
        assert chunk_config.markdown_split_headers is True

    def test_embedding_config_openai_settings(self):
        """Test OpenAI-specific embedding settings."""
        emb_config = config.EmbeddingConfig()

        # Test OpenAI defaults
        assert emb_config.openai_model == "text-embedding-3-small"
        assert emb_config.openai_api_key is None  # Not set by default
        assert emb_config.batch_size == 32
        assert emb_config.normalize is True

    def test_document_processor_create_chunk(self):
        """Test chunk creation helper."""
        doc_config = config.DocumentConfig()
        chunk_config = config.ChunkingConfig()
        processor = document_processor.DocumentProcessor(doc_config, chunk_config)

        # Test chunk creation
        chunk = processor._create_chunk(
            content="Test content", chunk_type="test", index=0, metadata_key="value"
        )
        assert chunk["content"] == "Test content"
        assert chunk["type"] == "test"
        assert chunk["metadata"]["index"] == 0
        assert chunk["metadata"]["metadata_key"] == "value"

    def test_redis_config_socket_settings(self):
        """Test Redis socket keepalive settings."""
        redis_config = config.RedisConfig()

        # Test socket settings
        assert redis_config.socket_keepalive is True
        assert isinstance(redis_config.socket_keepalive_options, dict)
        assert redis_config.max_connections == 50
        assert redis_config.decode_responses is False


if __name__ == "__main__":
    print("✅ Final tests to reach 80% coverage!")
