#!/usr/bin/env python
"""Fixed MCP Server launcher for Claude Desktop integration."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Create log directory in home
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
from pydantic import Field
from pydantic_settings import BaseSettings

from eol.rag_context.config import (
    CacheConfig,
    ChunkingConfig,
    DocumentConfig,
    EmbeddingConfig,
    IndexConfig,
    RedisConfig,
)
from eol.rag_context.document_processor import DocumentProcessor
from eol.rag_context.embeddings import EmbeddingManager
from eol.rag_context.indexer import DocumentIndexer

# Import individual components
from eol.rag_context.redis_client import RedisVectorStore


class MCPSafeRAGConfig(BaseSettings):
    """RAG configuration that works in MCP sandbox environment."""

    # Sub-configurations - use defaults
    redis: RedisConfig = Field(default_factory=RedisConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)

    # Use home directory paths that we know are writable
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".eol-rag-context" / "data")
    index_dir: Path = Field(default_factory=lambda: Path.home() / ".eol-rag-context" / "indexes")

    # Server settings
    server_name: str = Field(default="eol-rag-context")
    server_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)

    class Config:
        env_prefix = "RAG_"
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_directories(self):
        """Manually create directories after config is created."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


# Create MCP server
mcp = FastMCP("eol-rag-context")

# Global components
config: Optional[MCPSafeRAGConfig] = None
redis_store = None
embedding_manager = None
document_processor = None
indexer = None
initialized = False


async def initialize_components():
    """Initialize all RAG components."""
    global config, redis_store, embedding_manager, document_processor, indexer, initialized

    if initialized:
        return "Already initialized"

    try:
        logger.info("Initializing MCP RAG components...")

        # Create config without auto-creating directories
        config = MCPSafeRAGConfig()

        # Manually ensure directories exist
        config.ensure_directories()
        logger.info(f"Using data_dir: {config.data_dir}")
        logger.info(f"Using index_dir: {config.index_dir}")

        # Initialize Redis with both sync and async connections
        redis_store = RedisVectorStore(config.redis, config.index)
        redis_store.connect()  # Sync connection
        await redis_store.connect_async()  # Also need async connection for async methods
        redis_store.create_hierarchical_indexes(config.embedding.dimension)
        logger.info("Connected to Redis successfully")

        # Initialize other components
        embedding_manager = EmbeddingManager(config.embedding)
        document_processor = DocumentProcessor(config.document, config.chunking)

        # Create a custom config object that indexer expects
        full_config = type(
            "Config",
            (),
            {
                "redis": config.redis,
                "embedding": config.embedding,
                "index": config.index,
                "chunking": config.chunking,
                "cache": config.cache,
                "document": config.document,
                "data_dir": config.data_dir,
                "index_dir": config.index_dir,
            },
        )()

        indexer = DocumentIndexer(full_config, document_processor, embedding_manager, redis_store)

        initialized = True
        logger.info("All components initialized successfully")
        return "Components initialized successfully"

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}", exc_info=True)
        raise


@mcp.tool()
async def test_sandbox() -> dict:
    """Test sandbox environment and permissions.

    Returns:
        Environment information and write permissions
    """
    import os
    from pathlib import Path

    results = {
        "cwd": os.getcwd(),
        "user": os.getenv("USER", "unknown"),
        "home": str(Path.home()),
        "writable_paths": [],
        "read_only_paths": [],
    }

    # Test various paths
    test_paths = [
        ("home_config", Path.home() / ".eol-rag-context" / "test.txt"),
        ("tmp", Path("/tmp/eol-test.txt")),
        ("cwd", Path.cwd() / "test.txt"),
    ]

    for name, path in test_paths:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test")
            path.unlink()
            results["writable_paths"].append(str(path.parent))
        except Exception as e:
            results["read_only_paths"].append({"path": str(path.parent), "error": str(e)})

    return results


@mcp.tool()
async def index_directory(path: str, recursive: bool = True) -> dict:
    """Index a directory of documents.

    Args:
        path: Directory path to index
        recursive: Whether to index subdirectories

    Returns:
        Indexing statistics
    """
    try:
        await initialize_components()

        logger.info(f"Indexing directory: {path}")
        result = await indexer.index_folder(path, recursive=recursive)

        return {
            "status": "success",
            "source_id": result.source_id,
            "files_indexed": result.indexed_files,
            "chunks_created": result.total_chunks,
            "path": path,
        }
    except Exception as e:
        logger.error(f"Error indexing directory: {e}", exc_info=True)
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def search_context(query: str, max_results: int = 10) -> list:
    """Search for relevant context.

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        List of relevant documents
    """
    try:
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

    except Exception as e:
        logger.error(f"Error searching context: {e}", exc_info=True)
        return [{"error": str(e), "error_type": type(e).__name__}]


@mcp.tool()
async def list_sources() -> list:
    """List all indexed sources.

    Returns:
        List of indexed sources with metadata
    """
    try:
        await initialize_components()

        sources = await indexer.list_sources()  # Add await here

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
    except Exception as e:
        logger.error(f"Error listing sources: {e}", exc_info=True)
        return [{"error": str(e), "error_type": type(e).__name__}]


@mcp.tool()
async def remove_source(source_id: str) -> dict:
    """Remove an indexed source.

    Args:
        source_id: ID of the source to remove

    Returns:
        Removal status
    """
    try:
        await initialize_components()

        removed = await indexer.remove_source(source_id)  # Add await here

        return {
            "status": "success" if removed else "not_found",
            "source_id": source_id,
            "removed": removed,
        }
    except Exception as e:
        logger.error(f"Error removing source: {e}", exc_info=True)
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}


@mcp.tool()
async def get_stats() -> dict:
    """Get indexing statistics.

    Returns:
        Current statistics
    """
    try:
        await initialize_components()

        stats = indexer.get_stats()  # This is SYNC - no await
        sources = await indexer.list_sources()  # This is ASYNC - needs await

        return {
            "status": "success",
            "documents_indexed": stats.get("documents_indexed", 0),
            "chunks_created": stats.get("chunks_created", 0),
            "sources_count": len(sources),
            "redis_connected": redis_store.redis is not None,
            "data_dir": str(config.data_dir),
            "index_dir": str(config.index_dir),
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}


if __name__ == "__main__":
    logger.info("Starting MCP RAG Context Server (Fixed)")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Home directory: {Path.home()}")

    # Run the MCP server
    mcp.run()
