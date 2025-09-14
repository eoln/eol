"""Unit tests for batch operations module - high-performance batch processing."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from eol.rag_context.batch_operations import (
    BatchEmbeddingManager,
    BatchRedisClient,
    StreamingProcessor,
    batch_index_documents,
)
from eol.rag_context.redis_client import VectorDocument


class TestBatchEmbeddingManager:
    """Test BatchEmbeddingManager functionality."""

    @pytest.fixture
    def mock_embedding_manager(self):
        """Mock EmbeddingManager."""
        manager = AsyncMock()
        manager.get_embedding = AsyncMock()
        manager.get_embeddings = AsyncMock()
        return manager

    @pytest.fixture
    def batch_embedder(self, mock_embedding_manager):
        """Create BatchEmbeddingManager instance."""
        return BatchEmbeddingManager(mock_embedding_manager, max_batch_size=2)

    @pytest.mark.asyncio
    async def test_get_embeddings_batch_empty(self, batch_embedder):
        """Test batch embedding with empty input."""
        result = await batch_embedder.get_embeddings_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_get_embeddings_batch_with_cache(self, batch_embedder):
        """Test batch embedding with cache hits."""
        # Pre-populate cache
        cached_embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        batch_embedder._embedding_cache["cached_text"] = cached_embedding

        # Mock embedding for new text
        new_embedding = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        batch_embedder.embedding_manager.get_embeddings.return_value = [new_embedding]

        texts = ["cached_text", "new_text"]
        embeddings = await batch_embedder.get_embeddings_batch(texts)

        assert len(embeddings) == 2
        np.testing.assert_array_equal(embeddings[0], cached_embedding)
        np.testing.assert_array_equal(embeddings[1], new_embedding)

        # Should only call embedding manager for uncached text
        batch_embedder.embedding_manager.get_embeddings.assert_called_once_with(
            ["new_text"], use_cache=True
        )

    @pytest.mark.asyncio
    async def test_get_embeddings_batch_fallback(self, batch_embedder):
        """Test fallback to individual requests."""
        # Mock batch failure, individual success
        individual_embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        batch_embedder.embedding_manager.get_embeddings.side_effect = Exception("Batch failed")
        batch_embedder.embedding_manager.get_embedding.return_value = individual_embedding

        texts = ["text1", "text2"]
        embeddings = await batch_embedder.get_embeddings_batch(texts, use_cache=False)

        assert len(embeddings) == 2
        np.testing.assert_array_equal(embeddings[0], individual_embedding)
        np.testing.assert_array_equal(embeddings[1], individual_embedding)

        # Should call individual embedding for each text
        assert batch_embedder.embedding_manager.get_embedding.call_count == 2

    @pytest.mark.asyncio
    async def test_get_embeddings_batch_individual_fallback(self, batch_embedder):
        """Test fallback when individual embedding fails."""
        # Mock both batch and individual failures
        batch_embedder.embedding_manager.get_embeddings.side_effect = Exception("Batch failed")
        batch_embedder.embedding_manager.get_embedding.side_effect = Exception("Individual failed")

        texts = ["text1"]
        embeddings = await batch_embedder.get_embeddings_batch(texts, use_cache=False)

        assert len(embeddings) == 1
        # Should return zero vector as fallback
        assert embeddings[0].shape == (384,)
        assert np.allclose(embeddings[0], 0.0)

    def test_clear_cache(self, batch_embedder):
        """Test cache clearing."""
        batch_embedder._embedding_cache["test"] = np.array([1, 2, 3])
        batch_embedder.clear_cache()
        assert len(batch_embedder._embedding_cache) == 0

    def test_get_cache_stats(self, batch_embedder):
        """Test cache statistics."""
        batch_embedder._embedding_cache["test1"] = np.zeros(384, dtype=np.float32)
        batch_embedder._embedding_cache["test2"] = np.zeros(384, dtype=np.float32)

        stats = batch_embedder.get_cache_stats()
        assert stats["cache_size"] == 2
        assert stats["memory_estimate_mb"] > 0

    @pytest.mark.asyncio
    async def test_cache_size_limit(self, batch_embedder):
        """Test cache size limiting."""
        # Fill cache beyond limit
        for i in range(15000):  # More than 10000 limit
            batch_embedder._embedding_cache[f"text_{i}"] = np.zeros(384, dtype=np.float32)

        # Mock embedding generation
        new_embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        batch_embedder.embedding_manager.get_embeddings.return_value = [new_embedding]

        # This should trigger cache cleanup
        await batch_embedder.get_embeddings_batch(["new_text"], use_cache=True)

        # Cache should be reduced
        assert len(batch_embedder._embedding_cache) < 15000


class TestBatchRedisClient:
    """Test BatchRedisClient functionality."""

    @pytest.fixture
    def mock_redis_store(self):
        """Mock RedisVectorStore."""
        store = MagicMock()
        store.redis = MagicMock()
        store.redis.pipeline.return_value = MagicMock()
        store.async_redis = AsyncMock()
        store.index_config = MagicMock()
        store.index_config.concept_vectorset = "concepts"
        store.index_config.section_vectorset = "sections"
        store.index_config.chunk_vectorset = "chunks"
        store.index_config.vectorset_name = "default"
        return store

    @pytest.fixture
    def batch_client(self, mock_redis_store):
        """Create BatchRedisClient instance."""
        return BatchRedisClient(mock_redis_store, pipeline_size=2)

    @pytest.fixture
    def sample_documents(self):
        """Create sample VectorDocument objects."""
        docs = []
        for i in range(3):
            doc = VectorDocument(
                id=f"doc_{i}",
                content=f"Content {i}",
                embedding=np.random.rand(384).astype(np.float32),
                hierarchy_level=1,
                metadata={"source": f"file_{i}.py", "chunk_index": i},
            )
            docs.append(doc)
        return docs

    @pytest.mark.asyncio
    async def test_store_documents_batch_empty(self, batch_client):
        """Test storing empty document list."""
        result = await batch_client.store_documents_batch([])
        assert result == 0

    @pytest.mark.asyncio
    async def test_store_documents_batch(self, batch_client, sample_documents, mock_redis_store):
        """Test successful batch document storage."""
        # Mock pipeline execution
        mock_pipeline = MagicMock()
        mock_redis_store.redis.pipeline.return_value = mock_pipeline
        mock_pipeline.execute.return_value = [True] * len(sample_documents)

        # Mock VADD operations
        mock_redis_store.async_redis.execute_command = AsyncMock()

        result = await batch_client.store_documents_batch(sample_documents)

        # Should process all documents
        assert result == len(sample_documents)

        # Should have called pipeline operations
        assert mock_pipeline.hset.call_count == len(sample_documents)
        assert mock_redis_store.async_redis.execute_command.call_count == len(sample_documents)

    @pytest.mark.asyncio
    async def test_store_documents_batch_vadd_failure(
        self, batch_client, sample_documents, mock_redis_store
    ):
        """Test handling VADD operation failures."""
        # Mock pipeline success but VADD failure
        mock_pipeline = MagicMock()
        mock_redis_store.redis.pipeline.return_value = mock_pipeline
        mock_pipeline.execute.return_value = [True] * len(sample_documents)

        # Mock VADD failure
        mock_redis_store.async_redis.execute_command.side_effect = Exception("VADD failed")

        result = await batch_client.store_documents_batch(sample_documents)

        # No documents should be successfully stored due to VADD failures
        assert result == 0

    @pytest.mark.asyncio
    async def test_store_documents_batch_pipeline_failure(
        self, batch_client, sample_documents, mock_redis_store
    ):
        """Test fallback to individual storage on pipeline failure."""
        # Mock pipeline failure
        mock_redis_store.redis.pipeline.return_value.execute.side_effect = Exception(
            "Pipeline failed"
        )

        # Mock individual storage success
        mock_redis_store.store_document = AsyncMock()

        result = await batch_client.store_documents_batch(sample_documents)

        # Should fall back to individual storage
        assert mock_redis_store.store_document.call_count == len(sample_documents)
        assert result == len(sample_documents)

    def test_get_vectorset_name(self, batch_client):
        """Test vectorset name mapping."""
        assert batch_client._get_vectorset_name(1) == "concepts"
        assert batch_client._get_vectorset_name(2) == "sections"
        assert batch_client._get_vectorset_name(3) == "chunks"
        assert batch_client._get_vectorset_name(99) == "default"

    @pytest.mark.asyncio
    async def test_bulk_vector_search(self, batch_client, mock_redis_store):
        """Test bulk vector search functionality."""
        queries = ["query1", "query2"]

        # Mock embedding generation
        mock_embedding_manager = AsyncMock()
        query_embeddings = [np.random.rand(384).astype(np.float32) for _ in queries]

        with patch("eol.rag_context.batch_operations.BatchEmbeddingManager") as MockBatchEmbedder:
            mock_batch_embedder = MockBatchEmbedder.return_value
            mock_batch_embedder.get_embeddings_batch = AsyncMock(return_value=query_embeddings)

            # Mock search results
            search_results = [
                [("doc1", 0.9, {"content": "result1"})],
                [("doc2", 0.8, {"content": "result2"})],
            ]
            mock_redis_store.vector_search = AsyncMock(side_effect=search_results)

            results = await batch_client.bulk_vector_search(queries, mock_embedding_manager)

            assert len(results) == len(queries)
            assert results[0] == search_results[0]
            assert results[1] == search_results[1]

    @pytest.mark.asyncio
    async def test_bulk_vector_search_with_exception(self, batch_client, mock_redis_store):
        """Test bulk vector search exception handling."""
        queries = ["query1", "failing_query"]

        mock_embedding_manager = AsyncMock()
        query_embeddings = [np.random.rand(384).astype(np.float32) for _ in queries]

        with patch("eol.rag_context.batch_operations.BatchEmbeddingManager") as MockBatchEmbedder:
            mock_batch_embedder = MockBatchEmbedder.return_value
            mock_batch_embedder.get_embeddings_batch = AsyncMock(return_value=query_embeddings)

            # Mock search with alternating success/failure
            call_count = 0

            async def mock_search_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 2:  # Second call fails
                    raise Exception("Search failed")
                return [("doc1", 0.9, {"content": "result1"})]

            mock_redis_store.vector_search = AsyncMock(side_effect=mock_search_side_effect)

            results = await batch_client.bulk_vector_search(queries, mock_embedding_manager)

            assert len(results) == len(queries)
            # First result should be successful
            assert len(results[0]) > 0
            # Second result should be empty due to exception
            assert results[1] == []


class TestStreamingProcessor:
    """Test StreamingProcessor functionality."""

    @pytest.fixture
    def processor(self):
        """Create StreamingProcessor instance."""
        return StreamingProcessor(chunk_size=50)  # Small chunk for testing

    @pytest.mark.asyncio
    async def test_process_large_file_stream(self, processor, tmp_path):
        """Test streaming file processing."""
        # Create test file
        test_file = tmp_path / "test.txt"
        content = (
            "This is sentence one. This is sentence two. This is sentence three. Final sentence."
        )
        test_file.write_text(content)

        # Mock processor function
        processed_chunks = []

        async def mock_processor(text, chunk_id):
            processed_chunks.append({"text": text, "chunk_id": chunk_id})
            return f"processed_{chunk_id}"

        results = await processor.process_large_file_stream(str(test_file), mock_processor)

        # Should have processed file in chunks
        assert len(results) > 0
        assert len(processed_chunks) > 0

        # All chunks combined should contain the original content
        all_text = " ".join([chunk["text"] for chunk in processed_chunks])
        assert "sentence one" in all_text
        assert "Final sentence" in all_text

    @pytest.mark.asyncio
    async def test_process_large_file_stream_file_not_found(self, processor):
        """Test handling of non-existent file."""

        async def mock_processor(text, chunk_id):
            return f"processed_{chunk_id}"

        results = await processor.process_large_file_stream("/nonexistent/file.txt", mock_processor)

        # Should return empty results on file error
        assert results == []

    @pytest.mark.asyncio
    async def test_process_large_file_stream_processor_exception(self, processor, tmp_path):
        """Test handling of processor function exceptions."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content for exception handling.")

        # Mock processor that raises exception
        async def failing_processor(text, chunk_id):
            raise Exception("Processing failed")

        results = await processor.process_large_file_stream(str(test_file), failing_processor)

        # Should handle exceptions gracefully
        assert isinstance(results, list)


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_batch_index_documents(self):
        """Test batch_index_documents convenience function."""
        # Mock documents and redis store
        documents = [MagicMock() for _ in range(3)]
        mock_redis_store = MagicMock()

        with patch("eol.rag_context.batch_operations.BatchRedisClient") as MockBatchClient:
            mock_client = MockBatchClient.return_value
            mock_client.store_documents_batch = AsyncMock(return_value=3)

            result = await batch_index_documents(documents, mock_redis_store, batch_size=2)

            assert result == 3
            MockBatchClient.assert_called_once_with(mock_redis_store, pipeline_size=2)
            mock_client.store_documents_batch.assert_called_once_with(documents)
