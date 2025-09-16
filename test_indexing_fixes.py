#!/usr/bin/env python
"""Test script to validate indexing fixes."""

import asyncio
import logging
from pathlib import Path

# Set up logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


async def test_file_exclusion():
    """Test that .venv and .uv-cache directories are properly excluded."""
    from eol.rag_context.config import RAGConfig
    from eol.rag_context.indexer import FolderScanner

    config = RAGConfig()
    scanner = FolderScanner(config)

    # Test scanning current directory
    test_path = Path("/Users/pmarzec/Devel/eol/packages/eol-rag-context")

    print("\n=== Testing File Exclusion ===")
    print(f"Scanning: {test_path}")

    files = await scanner.scan_folder(test_path, recursive=True)

    # Check if any files from excluded directories are included
    excluded_found = []
    for file in files:
        for parent in file.parents:
            if parent.name in [".venv", ".uv-cache", "__pycache__", ".git"]:
                excluded_found.append(str(file))
                break

    if excluded_found:
        print(f"❌ Found {len(excluded_found)} files from excluded directories:")
        for f in excluded_found[:5]:  # Show first 5
            print(f"   - {f}")
    else:
        print("✅ No files from excluded directories found")

    print(f"Total files to index: {len(files)}")

    # Show sample of files that will be indexed
    print("\nSample files to be indexed:")
    for f in files[:10]:
        print(f"   - {f.relative_to(test_path)}")

    return len(excluded_found) == 0


async def test_vector_storage():
    """Test vector storage with proper dimensions."""
    import numpy as np

    from eol.rag_context.config import RAGConfig
    from eol.rag_context.redis_client import RedisVectorStore, VectorDocument

    print("\n=== Testing Vector Storage ===")

    config = RAGConfig()
    redis_config = config.redis
    index_config = config.index

    redis_store = RedisVectorStore(redis_config, index_config)

    # Create a test document with correct dimension
    test_doc = VectorDocument(
        id="test_doc_001",
        content="This is a test document",
        embedding=np.random.rand(384).astype(np.float32),  # 384 dimensions as configured
        hierarchy_level=3,  # Chunk level
        metadata={"test": True},
    )

    try:
        await redis_store.store_document(test_doc)
        print("✅ Successfully stored test document")
        return True
    except Exception as e:
        print(f"❌ Failed to store test document: {e}")
        return False


async def test_embedding_generation():
    """Test that embeddings are generated with correct dimensions."""
    from eol.rag_context.config import RAGConfig
    from eol.rag_context.embeddings import EmbeddingManager

    print("\n=== Testing Embedding Generation ===")

    config = RAGConfig()
    embedding_manager = EmbeddingManager(config.embedding)

    test_text = "This is a test text for embedding generation"

    try:
        embedding = await embedding_manager.get_embedding(test_text, use_cache=False)
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding dtype: {embedding.dtype}")

        if embedding.shape == (384,) or embedding.shape == (1, 384):
            print("✅ Embedding has correct dimensions")
            return True
        else:
            print(f"❌ Unexpected embedding dimensions: {embedding.shape}")
            return False
    except Exception as e:
        print(f"❌ Failed to generate embedding: {e}")
        return False


async def main():
    """Run all tests."""
    print("Testing Indexing Fixes")
    print("=" * 50)

    # Test file exclusion
    exclusion_ok = await test_file_exclusion()

    # Test embedding generation
    embedding_ok = await test_embedding_generation()

    # Test vector storage
    vector_ok = await test_vector_storage()

    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"  File Exclusion: {'✅ PASS' if exclusion_ok else '❌ FAIL'}")
    print(f"  Embedding Generation: {'✅ PASS' if embedding_ok else '❌ FAIL'}")
    print(f"  Vector Storage: {'✅ PASS' if vector_ok else '❌ FAIL'}")

    if exclusion_ok and embedding_ok and vector_ok:
        print("\n✅ All tests passed! Ready to restart MCP server.")
    else:
        print("\n❌ Some tests failed. Please review the errors above.")


if __name__ == "__main__":
    asyncio.run(main())
