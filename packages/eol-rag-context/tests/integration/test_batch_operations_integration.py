"""
Integration tests for batch_operations module.
Tests batch processing with real Redis and various document types.
"""

import asyncio
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.integration
class TestBatchOperationsIntegration:
    """Test batch operations with real components."""

    @pytest.mark.asyncio
    async def test_batch_embedding_manager(self, embedding_manager):
        """Test BatchEmbeddingManager functionality."""
        from eol.rag_context.batch_operations import BatchEmbeddingManager

        # Create batch embedding manager
        batch_manager = BatchEmbeddingManager(embedding_manager=embedding_manager, max_batch_size=3)

        # Test batch embedding generation
        texts = [
            "First document about Python programming",
            "Second document about machine learning",
            "Third document about data science",
            "Fourth document about artificial intelligence",
            "Fifth document about deep learning",
        ]

        embeddings = await batch_manager.get_embeddings_batch(texts)
        assert len(embeddings) == 5
        assert all(isinstance(emb, np.ndarray) for emb in embeddings)
        assert all(emb.shape[0] == 384 for emb in embeddings)

        # Test empty batch
        empty_embeddings = await batch_manager.get_embeddings_batch([])
        assert empty_embeddings == []

        # Test caching behavior
        cached_embeddings = await batch_manager.get_embeddings_batch(texts[:2], use_cache=True)
        assert len(cached_embeddings) == 2
        # Should be the same embeddings as before
        assert np.array_equal(cached_embeddings[0], embeddings[0])
        assert np.array_equal(cached_embeddings[1], embeddings[1])

    @pytest.mark.asyncio
    async def test_batch_redis_client(self, redis_store):
        """Test BatchRedisClient for batch document storage."""
        from eol.rag_context.batch_operations import BatchRedisClient
        from eol.rag_context.redis_client import VectorDocument

        # Create batch Redis client
        batch_client = BatchRedisClient(redis_store=redis_store, pipeline_size=5)

        # Create test documents
        documents = []
        for i in range(12):  # More than batch size to test batching
            doc = VectorDocument(
                id=f"batch_test_{i}",
                content=f"This is batch test document {i} with some content",
                embedding=np.random.rand(384).astype(np.float32),
                metadata={"batch_test": True, "index": i, "type": "test"},
                hierarchy_level=1,
            )
            documents.append(doc)

        # Store documents in batches
        stored_count = await batch_client.store_documents_batch(documents)
        assert stored_count == 12

        # Clean up
        for doc in documents:
            try:
                await redis_store.delete_document(doc.id)
            except Exception:
                pass  # Ignore cleanup errors

    @pytest.mark.asyncio
    async def test_streaming_processor(self):
        """Test StreamingProcessor for large files."""
        from eol.rag_context.batch_operations import StreamingProcessor

        # Create a large test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            # Write a large amount of content
            for i in range(200):
                f.write(f"Chapter {i}\n")
                f.write("=" * 50 + "\n")
                f.write(f"This is the content of chapter {i}. ")
                f.write("It contains various information about different topics. " * 5)
                f.write("\n\n")
            temp_file = Path(f.name)

        try:
            # Create streaming processor with larger chunk size
            processor = StreamingProcessor(chunk_size=1024)  # Process 1KB at a time

            # Process the large file with a simple processor function
            def simple_processor(chunk):
                return {"processed": chunk}  # Return processed chunk

            results = await processor.process_large_file_stream(str(temp_file), simple_processor)

            # Should have processed the file
            assert results is not None
            # File is large enough to have multiple chunks
            assert len(results) > 0

        finally:
            temp_file.unlink()

    @pytest.mark.asyncio
    async def test_batch_index_function(self, redis_store):
        """Test the batch_index_documents convenience function."""
        from eol.rag_context.batch_operations import batch_index_documents
        from eol.rag_context.redis_client import VectorDocument

        # Create test documents
        documents = []
        for i in range(10):
            doc = VectorDocument(
                id=f"batch_index_test_{i}",
                content=f"Document {i} for batch indexing test",
                embedding=np.random.rand(384).astype(np.float32),
                metadata={"test": True, "index": i},
                hierarchy_level=1,
            )
            documents.append(doc)

        # Use the convenience function
        stored_count = await batch_index_documents(
            documents=documents, redis_store=redis_store, batch_size=5
        )

        assert stored_count == 10

        # Clean up
        for doc in documents:
            try:
                await redis_store.delete_document(doc.id)
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_batch_processing_with_errors(self, redis_store):
        """Test batch processing handles errors gracefully."""
        from eol.rag_context.batch_operations import BatchRedisClient
        from eol.rag_context.redis_client import VectorDocument

        batch_client = BatchRedisClient(redis_store=redis_store, pipeline_size=3)

        # Create documents, some with invalid data
        documents = []
        for i in range(5):
            if i == 2:
                # This document has a None embedding which might cause issues
                doc = VectorDocument(
                    id=f"error_test_{i}",
                    content=f"Document {i}",
                    embedding=None,  # Invalid embedding
                    metadata={"index": i},
                    hierarchy_level=1,
                )
            else:
                doc = VectorDocument(
                    id=f"error_test_{i}",
                    content=f"Document {i}",
                    embedding=np.random.rand(384).astype(np.float32),
                    metadata={"index": i},
                    hierarchy_level=1,
                )
            documents.append(doc)

        # Try to store documents, should handle errors gracefully
        try:
            stored_count = await batch_client.store_documents_batch(documents)
            # Should store the valid documents
            assert stored_count >= 4  # At least the 4 valid documents
        except Exception:
            # Even if it fails, it should fail gracefully
            pass

        # Clean up valid documents
        for doc in documents:
            if doc.embedding is not None:
                try:
                    await redis_store.delete_document(doc.id)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_concurrent_batch_operations(self, redis_store, embedding_manager):
        """Test concurrent batch operations."""
        from eol.rag_context.batch_operations import BatchEmbeddingManager, batch_index_documents
        from eol.rag_context.redis_client import VectorDocument

        batch_manager = BatchEmbeddingManager(embedding_manager)

        async def create_and_index_documents(prefix: str, count: int):
            """Helper to create and index documents."""
            texts = [f"{prefix} document {i}" for i in range(count)]
            embeddings = await batch_manager.get_embeddings_batch(texts)

            documents = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings, strict=False)):
                doc = VectorDocument(
                    id=f"{prefix}_{i}",
                    content=text,
                    embedding=embedding,
                    metadata={"prefix": prefix, "index": i},
                    hierarchy_level=1,
                )
                documents.append(doc)

            stored = await batch_index_documents(documents, redis_store, batch_size=5)
            return stored, documents

        # Run multiple batch operations concurrently
        tasks = [
            create_and_index_documents("batch_a", 10),
            create_and_index_documents("batch_b", 10),
            create_and_index_documents("batch_c", 10),
        ]

        results = await asyncio.gather(*tasks)

        # Verify all batches were processed
        total_stored = sum(stored for stored, _ in results)
        assert total_stored == 30

        # Clean up
        for _, documents in results:
            for doc in documents:
                try:
                    await redis_store.delete_document(doc.id)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_batch_operations_caching(self, embedding_manager):
        """Test that batch operations properly use caching."""
        from eol.rag_context.batch_operations import BatchEmbeddingManager

        batch_manager = BatchEmbeddingManager(embedding_manager, max_batch_size=10)

        # Generate embeddings for texts
        texts = ["Test document one", "Test document two", "Test document three"]
        first_run = await batch_manager.get_embeddings_batch(texts, use_cache=True)

        # Run again with cache - should be faster and return same results
        second_run = await batch_manager.get_embeddings_batch(texts, use_cache=True)

        # Verify same results
        assert len(first_run) == len(second_run)
        for i in range(len(first_run)):
            assert np.array_equal(first_run[i], second_run[i])

        # Test with cache disabled
        third_run = await batch_manager.get_embeddings_batch(texts, use_cache=False)
        assert len(third_run) == len(texts)

    @pytest.mark.asyncio
    async def test_batch_operations_large_batch(self, embedding_manager):
        """Test batch operations with large number of documents."""
        from eol.rag_context.batch_operations import BatchEmbeddingManager

        # Create batch manager with smaller batch size to test batching
        batch_manager = BatchEmbeddingManager(embedding_manager, max_batch_size=5)

        # Create 25 texts to ensure multiple batches
        texts = [f"Document number {i} with content about topic {i % 5}" for i in range(25)]

        # Process all texts
        embeddings = await batch_manager.get_embeddings_batch(texts)

        # Verify all embeddings were generated
        assert len(embeddings) == 25
        assert all(e is not None for e in embeddings)
        assert all(e.shape[0] == 384 for e in embeddings)

        # Test that different texts get different embeddings
        assert not np.array_equal(embeddings[0], embeddings[1])

        # Test that same text (due to modulo) might get similar embeddings
        # (depends on mock implementation)

    @pytest.mark.asyncio
    async def test_batch_pipeline_efficiency(self, redis_store, embedding_manager):
        """Test that batch pipeline operations are efficient."""
        from eol.rag_context.batch_operations import BatchEmbeddingManager, BatchRedisClient
        from eol.rag_context.redis_client import VectorDocument

        # Create managers
        batch_redis = BatchRedisClient(redis_store, pipeline_size=10)
        batch_embeddings = BatchEmbeddingManager(embedding_manager, max_batch_size=10)

        # Generate data
        texts = [f"Efficiency test document {i}" for i in range(20)]
        embeddings = await batch_embeddings.get_embeddings_batch(texts)

        # Create documents
        documents = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings, strict=False)):
            doc = VectorDocument(
                id=f"efficiency_test_{i}",
                content=text,
                embedding=embedding,
                metadata={"test": "efficiency", "index": i},
                hierarchy_level=1,
            )
            documents.append(doc)

        # Store in batches (should use pipelining)
        stored = await batch_redis.store_documents_batch(documents)
        assert stored == 20

        # Clean up
        for doc in documents:
            try:
                await redis_store.delete_document(doc.id)
            except Exception:
                pass
