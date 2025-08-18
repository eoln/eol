#!/usr/bin/env python
"""Final fixed MCP Server launcher with extensive logging."""

import asyncio
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Create log directory in home
log_dir = Path.home() / ".eol-rag-context"
log_dir.mkdir(parents=True, exist_ok=True)

# Setup DETAILED logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum detail
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(funcName)s:%(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / f"mcp_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
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
    debug: bool = Field(default=True)  # Enable debug mode

    class Config:
        env_prefix = "RAG_"
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_directories(self):
        """Manually create directories after config is created."""
        logger.debug(f"Ensuring directory exists: {self.data_dir}")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensuring directory exists: {self.index_dir}")
        self.index_dir.mkdir(parents=True, exist_ok=True)


# Create MCP server
logger.info("Creating FastMCP server instance")
mcp = FastMCP("eol-rag-context")

# Global components
config: Optional[MCPSafeRAGConfig] = None
redis_store: Optional[RedisVectorStore] = None
embedding_manager: Optional[EmbeddingManager] = None
document_processor: Optional[DocumentProcessor] = None
indexer: Optional[DocumentIndexer] = None
initialized = False


async def initialize_components():
    """Initialize all RAG components with detailed logging."""
    global config, redis_store, embedding_manager, document_processor, indexer, initialized

    logger.debug(f"initialize_components called. initialized={initialized}")

    if initialized:
        logger.debug("Already initialized, returning early")
        return "Already initialized"

    try:
        logger.info("=" * 60)
        logger.info("Starting component initialization")
        logger.info("=" * 60)

        # Create config without auto-creating directories
        logger.debug("Creating MCPSafeRAGConfig...")
        config = MCPSafeRAGConfig()
        logger.debug(f"Config created. Debug mode: {config.debug}")

        # Manually ensure directories exist
        logger.debug("Ensuring directories exist...")
        config.ensure_directories()
        logger.info(f"✅ Using data_dir: {config.data_dir}")
        logger.info(f"✅ Using index_dir: {config.index_dir}")

        # Initialize Redis with both sync and async connections
        logger.debug("Initializing RedisVectorStore...")
        redis_store = RedisVectorStore(config.redis, config.index)

        logger.debug("Connecting to Redis (sync)...")
        redis_store.connect()
        logger.info("✅ Redis sync connection established")

        logger.debug("Connecting to Redis (async)...")
        await redis_store.connect_async()
        logger.info("✅ Redis async connection established")

        logger.debug("Creating hierarchical indexes...")
        redis_store.create_hierarchical_indexes(config.embedding.dimension)
        logger.info("✅ Hierarchical indexes created/verified")

        # Initialize other components
        logger.debug("Initializing EmbeddingManager...")
        embedding_manager = EmbeddingManager(config.embedding)
        logger.info("✅ EmbeddingManager initialized")

        logger.debug("Initializing DocumentProcessor...")
        document_processor = DocumentProcessor(config.document, config.chunking)
        logger.info("✅ DocumentProcessor initialized")

        # Create a custom config object that indexer expects
        logger.debug("Creating config object for DocumentIndexer...")
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

        logger.debug("Initializing DocumentIndexer...")
        indexer = DocumentIndexer(full_config, document_processor, embedding_manager, redis_store)
        logger.info("✅ DocumentIndexer initialized")

        initialized = True
        logger.info("=" * 60)
        logger.info("✅ ALL COMPONENTS INITIALIZED SUCCESSFULLY")
        logger.info("=" * 60)
        return "Components initialized successfully"

    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ INITIALIZATION FAILED")
        logger.error("=" * 60)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Stack trace:")
        logger.error(traceback.format_exc())
        raise


@mcp.tool()
async def test_sandbox() -> dict:
    """Test sandbox environment and permissions.

    Returns:
        Environment information and write permissions
    """
    logger.debug("test_sandbox called")

    results = {
        "timestamp": datetime.now().isoformat(),
        "cwd": os.getcwd(),
        "user": os.getenv("USER", "unknown"),
        "home": str(Path.home()),
        "python_version": sys.version,
        "writable_paths": [],
        "read_only_paths": [],
    }

    logger.debug(f"Environment: cwd={results['cwd']}, user={results['user']}")

    # Test various paths
    test_paths = [
        ("home_config", Path.home() / ".eol-rag-context" / "test.txt"),
        ("tmp", Path("/tmp/eol-test.txt")),
        ("cwd", Path.cwd() / "test.txt"),
    ]

    for name, path in test_paths:
        logger.debug(f"Testing write permission for {name}: {path}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test")
            path.unlink()
            results["writable_paths"].append(str(path.parent))
            logger.debug(f"✅ {name} is writable")
        except Exception as e:
            results["read_only_paths"].append({"path": str(path.parent), "error": str(e)})
            logger.debug(f"❌ {name} is read-only: {e}")

    logger.info(
        f"Sandbox test complete. Writable: {len(results['writable_paths'])}, Read-only: {len(results['read_only_paths'])}"
    )
    return results


@mcp.tool()
async def get_stats() -> dict:
    """Get indexing statistics with detailed logging.

    Returns:
        Current statistics
    """
    logger.debug("=" * 40)
    logger.debug("get_stats called")

    try:
        logger.debug("Calling initialize_components...")
        await initialize_components()

        logger.debug(f"Global indexer is: {indexer}")
        logger.debug(f"Global redis_store is: {redis_store}")

        if indexer is None:
            logger.error("❌ Indexer is None after initialization!")
            return {
                "status": "error",
                "error": "Indexer not initialized",
                "initialized": initialized,
            }

        # Get stats - this is SYNC
        logger.debug("Calling indexer.get_stats() (SYNC method)...")
        stats = indexer.get_stats()
        logger.debug(f"Stats returned: {stats}")

        # Get sources - this is ASYNC
        logger.debug("Calling indexer.list_sources() (ASYNC method)...")
        sources = await indexer.list_sources()
        logger.debug(f"Sources returned: {len(sources)} sources")

        result = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "documents_indexed": stats.get("documents_indexed", 0),
            "chunks_created": stats.get("chunks_created", 0),
            "sources_count": len(sources),
            "redis_connected": redis_store.redis is not None,
            "async_redis_connected": redis_store.async_redis is not None,
            "data_dir": str(config.data_dir),
            "index_dir": str(config.index_dir),
            "initialized": initialized,
        }

        logger.info(
            f"✅ Stats retrieved successfully: {result['documents_indexed']} docs, {result['chunks_created']} chunks"
        )
        return result

    except Exception as e:
        logger.error(f"❌ Error in get_stats: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }


@mcp.tool()
async def list_sources() -> list:
    """List all indexed sources with detailed logging.

    Returns:
        List of indexed sources with metadata
    """
    logger.debug("=" * 40)
    logger.debug("list_sources called")

    try:
        await initialize_components()

        if indexer is None:
            logger.error("❌ Indexer is None after initialization!")
            return [{"error": "Indexer not initialized", "initialized": initialized}]

        logger.debug("Calling indexer.list_sources() (ASYNC)...")
        sources = await indexer.list_sources()
        logger.debug(f"Retrieved {len(sources)} sources")

        result = []
        for source in sources:
            source_data = {
                "source_id": source.source_id,
                "path": source.path,
                "file_count": source.file_count,
                "chunks": source.total_chunks,
                "indexed_at": source.indexed_at,
            }
            result.append(source_data)
            logger.debug(f"Source: {source.source_id} - {source.file_count} files")

        logger.info(f"✅ Listed {len(result)} sources successfully")
        return result

    except Exception as e:
        logger.error(f"❌ Error in list_sources: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
        return [
            {"error": str(e), "error_type": type(e).__name__, "traceback": traceback.format_exc()}
        ]


@mcp.tool()
async def index_directory(path: str, recursive: bool = True) -> dict:
    """Index a directory of documents with detailed logging.

    Args:
        path: Directory path to index
        recursive: Whether to index subdirectories

    Returns:
        Indexing statistics
    """
    logger.debug("=" * 40)
    logger.debug(f"index_directory called with path={path}, recursive={recursive}")

    try:
        await initialize_components()

        if indexer is None:
            logger.error("❌ Indexer is None after initialization!")
            return {"status": "error", "error": "Indexer not initialized"}

        logger.info(f"📁 Starting to index directory: {path}")
        result = await indexer.index_folder(path, recursive=recursive)

        response = {
            "status": "success",
            "source_id": result.source_id,
            "files_indexed": result.indexed_files,
            "chunks_created": result.total_chunks,
            "path": path,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"✅ Indexed {response['files_indexed']} files, created {response['chunks_created']} chunks"
        )
        return response

    except Exception as e:
        logger.error(f"❌ Error in index_directory: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }


@mcp.tool()
async def search_context(query: str, max_results: int = 10) -> list:
    """Search for relevant context with detailed logging.

    Args:
        query: Search query
        max_results: Maximum number of results

    Returns:
        List of relevant documents
    """
    logger.debug("=" * 40)
    logger.debug(f"search_context called with query='{query}', max_results={max_results}")

    try:
        await initialize_components()

        if embedding_manager is None or redis_store is None:
            logger.error("❌ Components not initialized!")
            return [{"error": "Components not initialized"}]

        logger.debug("Getting query embedding...")
        query_embedding = await embedding_manager.get_embedding(query)
        logger.debug(f"Query embedding shape: {query_embedding.shape}")

        # Ensure async Redis is connected
        if not redis_store.async_redis:
            logger.debug("Async Redis not connected, connecting now...")
            await redis_store.connect_async()

        logger.debug("Performing vector search...")
        results = await redis_store.vector_search(query_embedding=query_embedding, k=max_results)
        logger.debug(f"Search returned {len(results)} results")

        # Format results
        formatted_results = []
        for i, (doc_id, score, data) in enumerate(results):
            formatted_results.append(
                {
                    "rank": i + 1,
                    "score": float(score),
                    "text": data.get("text", ""),
                    "source": data.get("source", "Unknown"),
                    "chunk_index": data.get("chunk_index", 0),
                }
            )
            logger.debug(f"Result {i+1}: score={score:.3f}, source={data.get('source', 'Unknown')}")

        logger.info(f"✅ Search complete: {len(formatted_results)} results for query '{query}'")
        return formatted_results

    except Exception as e:
        logger.error(f"❌ Error in search_context: {type(e).__name__}: {str(e)}")
        logger.error(traceback.format_exc())
        return [
            {"error": str(e), "error_type": type(e).__name__, "traceback": traceback.format_exc()}
        ]


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 Starting MCP RAG Context Server (Final Fixed Version)")
    logger.info("=" * 60)
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Home directory: {Path.home()}")
    log_filename = f"mcp_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.info(f"Log file: {log_dir / log_filename}")
    logger.info("=" * 60)

    # Run the MCP server
    mcp.run()
