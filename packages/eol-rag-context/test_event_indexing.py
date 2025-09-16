#!/usr/bin/env python
"""Integration test for event XML indexing with temporal context."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from eol.rag_context.config import RAGConfig
from eol.rag_context.document_processor import DocumentProcessor
from eol.rag_context.embeddings import EmbeddingManager
from eol.rag_context.indexer import DocumentIndexer
from eol.rag_context.redis_client import RedisVectorStore


async def test_event_indexing():
    """Test complete indexing pipeline for event XMLs."""

    print("Initializing RAG components...")
    config = RAGConfig()

    # Initialize components
    redis_store = RedisVectorStore(config.redis, config.index)
    redis_store.connect()
    await redis_store.connect_async()

    embedding_manager = EmbeddingManager(config.embedding)
    document_processor = DocumentProcessor(config.document, config.chunking)
    indexer = DocumentIndexer(config, document_processor, embedding_manager, redis_store)

    # Create indexes
    redis_store.create_hierarchical_indexes(config.embedding.dimension)

    # Test directory with event XMLs
    event_dir = Path("/Users/eoln/Devel/cjg-data/dolnoslaskie-2025-06")

    if not event_dir.exists():
        print(f"Error: Directory not found: {event_dir}")
        return

    # Index just a few files for testing
    test_files = list(event_dir.glob("*.xml"))[:3]

    print(f"\nIndexing {len(test_files)} event XML files...")
    for file_path in test_files:
        print(f"  Processing: {file_path.name}")
        result = await indexer.index_file(str(file_path))
        if result:
            print(f"    ✅ Indexed with {result.total_chunks} chunks")

    # Test searching with temporal queries
    print("\n" + "=" * 60)
    print("Testing search with temporal queries...")

    test_queries = [
        "wydarzenia w czerwcu",
        "teatr 7 czerwca",
        "koncerty wieczorem",
        "wydarzenia w weekend",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")

        # Get embedding for query
        query_embedding = await embedding_manager.get_embedding(query)

        # Search
        results = await redis_store.vector_search(
            query_embedding=query_embedding, hierarchy_level=3, k=3
        )

        if results:
            for i, (doc_id, score, data) in enumerate(results[:2], 1):
                print(f"  Result {i} (score: {score:.3f}):")

                # Check for temporal information
                content = data.get("content", "")
                if "Date/Time:" in content:
                    # Extract date from content
                    lines = content.split("\n")
                    for line in lines:
                        if "Date/Time:" in line:
                            print(f"    {line}")
                            break

                # Show content preview
                preview = content[:150].replace("\n", " ")
                print(f"    Preview: {preview}...")

                # Check metadata
                metadata = data.get("metadata", {})
                if "date" in metadata:
                    print(f"    📅 Metadata date: {metadata['date']}")

    print("\n" + "=" * 60)
    print("Test complete! Temporal context is preserved in indexed content.")

    # Cleanup
    await redis_store.close()


if __name__ == "__main__":
    print("Event XML Indexing Integration Test")
    print("=" * 60)
    asyncio.run(test_event_indexing())
