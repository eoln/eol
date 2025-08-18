#!/usr/bin/env python
"""Simple MCP server runner for testing."""

import asyncio
import logging
from pathlib import Path

from eol.rag_context.config import RAGConfig
from eol.rag_context.document_processor import DocumentProcessor
from eol.rag_context.embeddings import EmbeddingManager
from eol.rag_context.indexer import DocumentIndexer
from eol.rag_context.redis_client import RedisVectorStore
from eol.rag_context.semantic_cache import SemanticCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Run MCP server with basic configuration."""

    # Create basic config
    config = RAGConfig()
    config.debug = True

    # Initialize components
    logger.info("Initializing Redis connection...")
    redis_store = RedisVectorStore(config.redis, config.index)
    redis_store.connect()  # Use sync connection for simplicity

    logger.info("Creating hierarchical indexes...")
    redis_store.create_hierarchical_indexes(config.embedding.dimension)

    logger.info("Initializing embedding manager...")
    embedding_manager = EmbeddingManager(config.embedding)

    logger.info("Initializing document processor...")
    document_processor = DocumentProcessor(config.document, config.chunking)

    logger.info("Initializing document indexer...")
    indexer = DocumentIndexer(config, document_processor, embedding_manager, redis_store)

    logger.info("Initializing semantic cache...")
    cache = SemanticCache(config.cache, embedding_manager, redis_store)

    # Index some sample documents
    test_dir = Path("./examples")
    if test_dir.exists():
        logger.info(f"Indexing sample documents from {test_dir}...")
        result = await indexer.index_folder(str(test_dir), recursive=False)
        logger.info(f"Indexed {result.indexed_files} files, {result.total_chunks} chunks")

    # Test search
    logger.info("\n=== Testing search functionality ===")
    query = "How to use RAG context?"
    query_embedding = await embedding_manager.get_embedding(query)
    results = await redis_store.vector_search(query_embedding=query_embedding, k=5)

    logger.info(f"Query: {query}")
    logger.info(f"Found {len(results)} results:")
    for i, (doc_id, score, data) in enumerate(results, 1):
        source = data.get("source", "Unknown")
        text_preview = data.get("text", "")[:100]
        logger.info(f"  {i}. Score: {score:.3f} - {source}")
        logger.info(f"     Preview: {text_preview}...")

    logger.info("\n✅ MCP server components initialized successfully!")
    logger.info("You can now:")
    logger.info("1. Use the indexer to index documents: indexer.index_folder('/path/to/docs')")
    logger.info("2. Search for content: redis_store.vector_search(query_embedding, k=5)")
    logger.info(
        "3. Use semantic cache: await cache.get('query') or await cache.set('query', 'response')"
    )

    logger.info("\n🎉 Test completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
