#!/usr/bin/env python
"""MCP Server launcher for Claude Desktop integration."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Create log directory
log_dir = Path.home() / ".eol-rag-context"
log_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "mcp_server.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastmcp import FastMCP

from eol.rag_context.config import RAGConfig
from eol.rag_context.document_processor import DocumentProcessor
from eol.rag_context.embeddings import EmbeddingManager
from eol.rag_context.indexer import DocumentIndexer
from eol.rag_context.redis_client import RedisVectorStore

# Create MCP server
mcp = FastMCP("eol-rag-context")

# Global components
config = None
redis_store = None
embedding_manager = None
document_processor = None
indexer = None
initialized = False


async def initialize_components():
    """Initialize all RAG components."""
    global config, redis_store, embedding_manager, document_processor, indexer, initialized

    if initialized:
        return

    # Create config with writable directories in user's home
    import os
    from pathlib import Path

    # Set environment variables BEFORE creating config
    # This way the config will use these paths instead of defaults
    home_dir = Path.home() / ".eol-rag-context"
    os.environ["RAG_DATA_DIR"] = str(home_dir / "data")
    os.environ["RAG_INDEX_DIR"] = str(home_dir / "indexes")

    # Ensure directories exist first
    (home_dir / "data").mkdir(parents=True, exist_ok=True)
    (home_dir / "indexes").mkdir(parents=True, exist_ok=True)

    # Now create config - it will use the env vars
    config = RAGConfig()

    # Initialize Redis with async connection
    try:
        redis_store = RedisVectorStore(config.redis, config.index)
        redis_store.connect()  # Use sync for now
        redis_store.create_hierarchical_indexes(config.embedding.dimension)
        logger.info("Successfully connected to Redis")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise

    # Initialize other components
    embedding_manager = EmbeddingManager(config.embedding)
    document_processor = DocumentProcessor(config.document, config.chunking)
    indexer = DocumentIndexer(config, document_processor, embedding_manager, redis_store)

    initialized = True
    return "Components initialized successfully"


@mcp.tool()
async def index_directory(path: str, recursive: bool = True) -> dict:
    """Index a directory of documents.

    Args:
        path: Directory path to index
        recursive: Whether to index subdirectories

    Returns:
        Indexing statistics
    """
    await initialize_components()

    result = await indexer.index_folder(path, recursive=recursive)

    return {
        "status": "success",
        "source_id": result.source_id,
        "files_indexed": result.indexed_files,
        "chunks_created": result.total_chunks,
        "path": path,
    }


@mcp.tool()
async def search_context(query: str, max_results: int = 10) -> list:
    """Search for relevant context.

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        List of relevant documents
    """
    await initialize_components()

    # Get embedding for query
    query_embedding = await embedding_manager.get_embedding(query)

    # Initialize async Redis if needed
    if not redis_store.async_redis:
        await redis_store.connect_async()

    # Search
    results = await redis_store.vector_search(query_embedding=query_embedding, k=max_results)

    # Format results
    formatted_results = []
    for doc_id, score, data in results:
        formatted_results.append(
            {
                "score": float(score),
                "text": data.get("text", ""),
                "source": data.get("source", "Unknown"),
                "chunk_index": data.get("chunk_index", 0),
            }
        )

    return formatted_results


@mcp.tool()
async def list_sources() -> list:
    """List all indexed sources.

    Returns:
        List of indexed sources with metadata
    """
    await initialize_components()

    sources = indexer.list_sources()

    return [
        {
            "source_id": source.source_id,
            "path": source.path,
            "file_count": source.file_count,
            "chunks": source.total_chunks,
            "indexed_at": source.indexed_at,
        }
        for source in sources
    ]


@mcp.tool()
async def remove_source(source_id: str) -> dict:
    """Remove an indexed source.

    Args:
        source_id: ID of the source to remove

    Returns:
        Removal status
    """
    await initialize_components()

    removed = indexer.remove_source(source_id)

    return {
        "status": "success" if removed else "not_found",
        "source_id": source_id,
        "removed": removed,
    }


@mcp.tool()
async def get_stats() -> dict:
    """Get indexing statistics.

    Returns:
        Current statistics
    """
    try:
        await initialize_components()

        stats = indexer.get_stats()

        return {
            "status": "success",
            "documents_indexed": stats.get("documents_indexed", 0),
            "chunks_created": stats.get("chunks_created", 0),
            "sources_count": len(indexer.list_sources()),
            "redis_connected": redis_store.redis is not None,
            "data_dir": str(config.data_dir),
            "index_dir": str(config.index_dir),
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
