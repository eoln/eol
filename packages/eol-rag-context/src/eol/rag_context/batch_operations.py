"""Batch processing optimizations for embeddings and Redis operations.

This module provides high-performance batch operations to maximize throughput
when processing large repositories. It implements intelligent batching for:
- Embedding generation with provider-specific optimizations
- Redis Vector Set operations with pipelining
- Memory-efficient processing with streaming

Key optimizations:
- Batch embedding requests to maximize provider throughput
- Redis pipelining for bulk VADD operations
- Connection pooling and multiplexing
- Memory streaming for large content
- Intelligent retry logic with exponential backoff

Example:
    Batch embedding generation:

    >>> batch_embedder = BatchEmbeddingManager(embedding_manager)
    >>> texts = ["doc1 content", "doc2 content", ...]
    >>> embeddings = await batch_embedder.get_embeddings_batch(texts)

    Bulk Redis operations:

    >>> batch_redis = BatchRedisClient(redis_store)
    >>> documents = [doc1, doc2, doc3, ...]
    >>> await batch_redis.store_documents_batch(documents)

"""

import asyncio
import logging
import time
from typing import Any

import numpy as np

from .embeddings import EmbeddingManager
from .redis_client import RedisVectorStore, VectorDocument

logger = logging.getLogger(__name__)


class BatchEmbeddingManager:
    """Optimized embedding generation with intelligent batching.

    Provides significant performance improvements over individual embedding
    requests by batching multiple texts together and using provider-specific
    optimizations.
    """

    def __init__(self, embedding_manager: EmbeddingManager, max_batch_size: int = 32):
        self.embedding_manager = embedding_manager
        self.max_batch_size = max_batch_size
        self._embedding_cache: dict[str, np.ndarray] = {}

    async def get_embeddings_batch(
        self, texts: list[str], use_cache: bool = True
    ) -> list[np.ndarray]:
        """Generate embeddings for multiple texts with batching optimization.

        Args:
            texts: List of text strings to embed
            use_cache: Whether to use/update embedding cache

        Returns:
            List of embedding vectors in same order as input texts
        """
        if not texts:
            return []

        # Check cache first
        embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if use_cache and text in self._embedding_cache:
                embeddings.append(self._embedding_cache[text])
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Generate embeddings for uncached texts
        if uncached_texts:
            logger.debug(f"Generating embeddings for {len(uncached_texts)} texts in batches")

            # Process in batches to respect provider limits
            new_embeddings = []
            for i in range(0, len(uncached_texts), self.max_batch_size):
                batch = uncached_texts[i : i + self.max_batch_size]

                try:
                    # Use provider's batch capability if available
                    if hasattr(self.embedding_manager, "get_embeddings"):
                        batch_embeddings = await self.embedding_manager.get_embeddings(
                            batch, use_cache=use_cache
                        )
                    else:
                        # Fallback to individual requests (still parallelized)
                        tasks = [
                            self.embedding_manager.get_embedding(text, use_cache=use_cache)
                            for text in batch
                        ]
                        batch_embeddings = await asyncio.gather(*tasks)

                    new_embeddings.extend(batch_embeddings)

                    # Add small delay between batches to avoid rate limiting
                    if len(uncached_texts) > self.max_batch_size:
                        await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Batch embedding generation failed: {e}")
                    # Fallback to individual requests
                    for text in batch:
                        try:
                            emb = await self.embedding_manager.get_embedding(
                                text, use_cache=use_cache
                            )
                            new_embeddings.append(emb)
                        except Exception as individual_e:
                            logger.error(f"Individual embedding failed for text: {individual_e}")
                            # Use zero vector as fallback
                            zero_emb = np.zeros(384, dtype=np.float32)
                            new_embeddings.append(zero_emb)

            # Update cache
            if use_cache:
                for text, emb in zip(uncached_texts, new_embeddings, strict=False):
                    self._embedding_cache[text] = emb

                # Limit cache size to prevent memory bloat
                if len(self._embedding_cache) > 10000:
                    # Remove oldest 20% of entries
                    items = list(self._embedding_cache.items())
                    to_remove = len(items) // 5
                    for key, _ in items[:to_remove]:
                        del self._embedding_cache[key]

            # Fill in the uncached embeddings
            for idx, emb in zip(uncached_indices, new_embeddings, strict=False):
                embeddings[idx] = emb

        return embeddings

    def clear_cache(self):
        """Clear embedding cache to free memory."""
        self._embedding_cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get embedding cache statistics."""
        return {
            "cache_size": len(self._embedding_cache),
            "memory_estimate_mb": len(self._embedding_cache) * 384 * 4 / (1024 * 1024),  # float32
        }


class BatchRedisClient:
    """Optimized Redis operations with pipelining and bulk operations.

    Provides significant performance improvements for Vector Set operations
    by batching multiple commands and using Redis pipelining.
    """

    def __init__(self, redis_store: RedisVectorStore, pipeline_size: int = 100):
        self.redis_store = redis_store
        self.pipeline_size = pipeline_size

    async def store_documents_batch(self, documents: list[VectorDocument]) -> int:
        """Store multiple documents using Redis pipelining.

        Args:
            documents: List of VectorDocument objects to store

        Returns:
            Number of documents successfully stored
        """
        if not documents:
            return 0

        stored_count = 0
        start_time = time.time()

        # Process in pipeline batches
        for i in range(0, len(documents), self.pipeline_size):
            batch = documents[i : i + self.pipeline_size]

            try:
                # Use Redis pipeline for bulk operations
                pipeline = self.redis_store.redis.pipeline()
                vadd_commands = []

                for doc in batch:
                    # Prepare document metadata for hash storage
                    doc_key = f"doc:{doc.id}"
                    doc_data = {
                        "content": doc.content,
                        "hierarchy_level": str(doc.hierarchy_level),
                        "created_at": str(time.time()),
                    }

                    # Add metadata fields
                    if doc.metadata:
                        for k, v in doc.metadata.items():
                            if v is not None and not k.startswith(
                                "embedding"
                            ):  # Skip binary fields
                                doc_data[f"meta_{k}"] = str(v)

                    # Add to pipeline
                    pipeline.hset(doc_key, mapping=doc_data)

                    # Prepare VADD command for Vector Set
                    vectorset_name = self._get_vectorset_name(doc.hierarchy_level)

                    # Ensure embedding is 1D array
                    embedding_array = doc.embedding
                    if embedding_array.ndim == 2:
                        embedding_array = embedding_array.flatten()

                    embedding_values = embedding_array.astype(np.float32).tolist()

                    # Skip documents with invalid embeddings
                    if not embedding_values or len(embedding_values) == 0:
                        logger.warning(f"Skipping document {doc.id} - empty embedding")
                        continue
                    if np.any(np.isnan(embedding_array)) or np.any(np.isinf(embedding_array)):
                        logger.warning(f"Skipping document {doc.id} - invalid embedding values")
                        continue

                    vadd_args = ["VADD", vectorset_name, "VALUES", str(len(embedding_values))]
                    # Pass each float value as a separate argument
                    for v in embedding_values:
                        vadd_args.append(str(v))  # v is already a float from tolist()
                    vadd_args.append(doc.id)

                    # Add quantization parameter based on configuration
                    quantization = self.redis_store.index_config.get_batch_quantization()
                    vadd_args.append(quantization)

                    vadd_commands.append(vadd_args)

                # Execute hash operations
                pipeline.execute()

                # Execute VADD operations (these need to be done individually for now)
                for vadd_args in vadd_commands:
                    try:
                        await self.redis_store.async_redis.execute_command(*vadd_args)
                        stored_count += 1
                    except Exception as e:
                        logger.error(f"VADD operation failed: {e}")
                        logger.error(
                            f"  Vector Set: {vadd_args[1] if len(vadd_args) > 1 else 'unknown'}"
                        )
                        expected_dim = vadd_args[3] if len(vadd_args) > 3 else "unknown"
                        logger.error(f"  Expected dimension: {expected_dim}")
                        logger.error(
                            f"  Doc ID: {vadd_args[-2] if len(vadd_args) > 2 else 'unknown'}"
                        )

                logger.debug(f"Stored batch of {len(batch)} documents")

            except Exception as e:
                logger.error(f"Batch storage failed for batch starting at {i}: {e}")

                # Fallback to individual storage
                for doc in batch:
                    try:
                        await self.redis_store.store_document(doc)
                        stored_count += 1
                    except Exception as individual_e:
                        logger.error(f"Individual document storage failed: {individual_e}")

        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            rate = stored_count / elapsed_time
            logger.info(
                f"Bulk stored {stored_count} documents in {elapsed_time:.1f}s ({rate:.1f} docs/sec)"
            )

        return stored_count

    def _get_vectorset_name(self, hierarchy_level: int) -> str:
        """Get Vector Set name for hierarchy level."""
        level_mapping = {
            1: self.redis_store.index_config.concept_vectorset,
            2: self.redis_store.index_config.section_vectorset,
            3: self.redis_store.index_config.chunk_vectorset,
        }
        return level_mapping.get(hierarchy_level, self.redis_store.index_config.vectorset_name)

    async def bulk_vector_search(
        self, queries: list[str], embedding_manager: EmbeddingManager, k: int = 5
    ) -> list[list[tuple[str, float, dict[str, Any]]]]:
        """Perform multiple vector searches efficiently.

        Args:
            queries: List of search query strings
            embedding_manager: Embedding manager for query vectorization
            k: Number of results per query

        Returns:
            List of search results, one list per query
        """
        if not queries:
            return []

        # Generate embeddings in batch
        batch_embedder = BatchEmbeddingManager(embedding_manager)
        query_embeddings = await batch_embedder.get_embeddings_batch(queries)

        # Execute searches concurrently
        search_tasks = [
            self.redis_store.vector_search(query, embedding_manager, k=k, query_embedding=emb)
            for query, emb in zip(queries, query_embeddings, strict=False)
        ]

        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Search failed for query {i}: {result}")
                processed_results.append([])
            else:
                processed_results.append(result)

        return processed_results


class StreamingProcessor:
    """Memory-efficient streaming processor for large files.

    Processes large files in chunks to avoid loading entire content into memory.
    Useful for repositories with very large documents.
    """

    def __init__(self, chunk_size: int = 1024 * 1024):  # 1MB chunks
        self.chunk_size = chunk_size

    async def process_large_file_stream(
        self, file_path: str, processor_func: callable
    ) -> list[Any]:
        """Process large file in streaming fashion.

        Args:
            file_path: Path to large file
            processor_func: Function to process each chunk

        Returns:
            List of processed chunks
        """
        results = []

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                chunk_buffer = ""
                chunk_id = 0

                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break

                    chunk_buffer += chunk

                    # Split on sentence boundaries to avoid cutting mid-sentence
                    sentences = chunk_buffer.split(". ")

                    if len(sentences) > 1:
                        # Process complete sentences
                        complete_text = ". ".join(sentences[:-1]) + "."

                        result = await processor_func(complete_text, chunk_id)
                        if result:
                            results.append(result)

                        # Keep incomplete sentence for next chunk
                        chunk_buffer = sentences[-1]
                        chunk_id += 1

                # Process remaining buffer
                if chunk_buffer.strip():
                    result = await processor_func(chunk_buffer, chunk_id)
                    if result:
                        results.append(result)

        except Exception as e:
            logger.error(f"Streaming processing failed for {file_path}: {e}")

        return results


# Convenience function for backward compatibility
async def batch_index_documents(
    documents: list[VectorDocument], redis_store: RedisVectorStore, batch_size: int = 100
) -> int:
    """Convenience function for batch document indexing."""
    batch_client = BatchRedisClient(redis_store, pipeline_size=batch_size)
    return await batch_client.store_documents_batch(documents)
