"""EOL RAG Context MCP Server."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastmcp import Context, FastMCP
from pydantic import BaseModel, Field

from .config import RAGConfig
from .document_processor import DocumentProcessor
from .embeddings import EmbeddingManager
from .file_watcher import FileWatcher
from .indexer import DocumentIndexer
from .knowledge_graph import KnowledgeGraphBuilder
from .redis_client import RedisVectorStore
from .semantic_cache import SemanticCache
from .async_task_manager import AsyncTaskManager, TaskStatus
from .parallel_indexer import ParallelIndexer, ParallelIndexingConfig

logger = logging.getLogger(__name__)


# Pydantic models for MCP tools
class StartIndexingRequest(BaseModel):
    """Request to start asynchronous indexing."""
    
    path: str = Field(description="Directory path to index")
    recursive: bool = Field(default=True, description="Index subdirectories")
    force_reindex: bool = Field(default=False, description="Force reindex of unchanged files")
    watch: bool = Field(default=False, description="Watch for changes after indexing")
    
    # Parallel processing configuration
    max_workers: int = Field(default=16, description="Maximum concurrent workers")
    batch_size: int = Field(default=32, description="Batch size for processing")
    enable_streaming: bool = Field(default=True, description="Enable streaming for large files")


class IndexingStatusRequest(BaseModel):
    """Request to get indexing task status."""
    
    task_id: str = Field(description="Task ID to check status for")


class ListTasksRequest(BaseModel):
    """Request to list indexing tasks."""
    
    status_filter: str | None = Field(default=None, description="Filter by status (pending, running, completed, failed, cancelled)")
    limit: int = Field(default=50, description="Maximum tasks to return")


class CancelTaskRequest(BaseModel):
    """Request to cancel an indexing task."""
    
    task_id: str = Field(description="Task ID to cancel")


class SearchContextRequest(BaseModel):
    """Request to search for context."""

    query: str = Field(description="Search query")
    max_results: int = Field(default=10, description="Maximum results to return")
    min_relevance: float = Field(default=0.7, description="Minimum relevance score")
    hierarchy_level: int | None = Field(
        default=None,
        description="Search at specific hierarchy level (1=concept, 2=section, 3=chunk)",
    )
    source_filter: str | None = Field(default=None, description="Filter by source ID")


class QueryKnowledgeGraphRequest(BaseModel):
    """Request to query knowledge graph."""

    query: str = Field(description="Query for knowledge graph")
    max_depth: int = Field(default=2, description="Maximum traversal depth")
    max_entities: int = Field(default=20, description="Maximum entities to return")


class OptimizeContextRequest(BaseModel):
    """Request to optimize context for LLM."""

    query: str = Field(description="User query")
    current_context: str | None = Field(default=None, description="Current context to optimize")
    max_tokens: int = Field(default=32000, description="Maximum context tokens")
    strategy: str = Field(
        default="hierarchical",
        description="Context strategy (hierarchical, flat, semantic)",
    )


class WatchDirectoryRequest(BaseModel):
    """Request to watch a directory for changes."""

    path: str = Field(description="Directory path to watch")
    recursive: bool = Field(default=True, description="Watch subdirectories")
    file_patterns: list[str] | None = Field(default=None, description="File patterns to watch")


class EOLRAGContextServer:
    """MCP server for intelligent RAG-based context management.

    This server implements the Model Context Protocol (MCP) to provide dynamic,
    intelligent context retrieval using Redis-backed vector storage. It replaces
    static documentation with semantic search across hierarchically indexed
    documents.

    The server provides:
    - Hierarchical document indexing (concepts → sections → chunks)
    - Vector similarity search with Redis Stack
    - Knowledge graph construction and querying
    - Real-time file watching and automatic reindexing
    - Semantic caching with adaptive hit rate optimization
    - Multi-format document processing (code, markdown, PDFs, etc.)

    Attributes:
        config: Configuration settings for the server.
        mcp: FastMCP server instance for protocol handling.
        redis_store: Vector storage backend using Redis Stack.
        embedding_manager: Handles embedding generation and caching.
        document_processor: Processes documents into structured chunks.
        indexer: Manages document indexing and hierarchical organization.
        semantic_cache: Caches query results for improved performance.
        knowledge_graph: Builds and queries entity relationship graphs.
        file_watcher: Monitors file changes for automatic reindexing.

    Example:
        Basic server setup:

        >>> from eol.rag_context import EOLRAGContextServer
        >>> from eol.rag_context.config import RAGConfig
        >>>
        >>> # Initialize with default configuration
        >>> server = EOLRAGContextServer()
        >>> await server.initialize()
        >>>
        >>> # Index documents
        >>> result = await server.index_directory("/path/to/docs")
        >>> print(f"Indexed {result['indexed_files']} files")

        Custom configuration:

        >>> config = RAGConfig(
        ...     redis_host="localhost",
        ...     embedding_provider="sentence-transformers",
        ...     cache_enabled=True
        ... )
        >>> server = EOLRAGContextServer(config)
        >>> await server.initialize()

    """

    def __init__(self, config: RAGConfig | None = None):
        self.config = config or RAGConfig()
        self.mcp = FastMCP(name=self.config.server_name, version=self.config.server_version)

        # Core components
        self.redis_store: RedisVectorStore | None = None
        self.embedding_manager: EmbeddingManager | None = None
        self.document_processor: DocumentProcessor | None = None
        self.indexer: DocumentIndexer | None = None
        self.semantic_cache: SemanticCache | None = None
        self.knowledge_graph: KnowledgeGraphBuilder | None = None
        self.file_watcher: FileWatcher | None = None
        
        # Non-blocking indexing components
        self.task_manager: AsyncTaskManager | None = None
        self.parallel_indexer: ParallelIndexer | None = None

        # Setup MCP handlers
        self._setup_resources()
        self._setup_tools()
        self._setup_prompts()

    async def initialize(self) -> None:
        """Initialize all server components in proper dependency order.

        This method sets up the complete RAG pipeline including Redis connection,
        embedding providers, document processing, indexing, caching, knowledge graph,
        and file watching capabilities. Components are initialized in dependency order
        to ensure proper functionality.

        The initialization process:
        1. Establishes Redis connection and creates vector indexes
        2. Initializes embedding manager with configured provider
        3. Sets up document processor for multi-format support
        4. Creates document indexer with hierarchical organization
        5. Initializes semantic cache for query optimization
        6. Sets up knowledge graph builder for entity relationships
        7. Starts file watcher for real-time updates

        Raises:
            RedisConnectionError: If unable to connect to Redis Stack.
            EmbeddingProviderError: If embedding provider initialization fails.
            ConfigurationError: If required configuration is missing or invalid.

        Example:
            >>> server = EOLRAGContextServer()
            >>> await server.initialize()
            >>> print("Server ready for indexing and search")
            Server ready for indexing and search

        """
        logger.info(f"Initializing {self.config.server_name} v{self.config.server_version}")

        # Initialize Redis
        self.redis_store = RedisVectorStore(self.config.redis, self.config.index)
        self.redis_store.connect()  # Initialize sync connection for background operations
        await self.redis_store.connect_async()  # Initialize async connection for async operations

        # Initialize embeddings
        self.embedding_manager = EmbeddingManager(
            self.config.embedding, self.redis_store.async_redis
        )

        # Create indexes
        self.redis_store.create_hierarchical_indexes(self.config.embedding.dimension)

        # Initialize processors
        self.document_processor = DocumentProcessor(self.config.document, self.config.chunking)

        self.indexer = DocumentIndexer(
            self.config,
            self.document_processor,
            self.embedding_manager,
            self.redis_store,
        )

        # Initialize semantic cache
        self.semantic_cache = SemanticCache(
            self.config.cache, self.embedding_manager, self.redis_store
        )
        await self.semantic_cache.initialize()

        # Initialize knowledge graph
        self.knowledge_graph = KnowledgeGraphBuilder(self.redis_store, self.embedding_manager)

        # Initialize file watcher
        self.file_watcher = FileWatcher(self.indexer, self.knowledge_graph, debounce_seconds=2.0)
        await self.file_watcher.start()

        # Initialize non-blocking indexing components
        self.task_manager = AsyncTaskManager(self.redis_store)
        self.parallel_indexer = ParallelIndexer(
            self.config,
            self.document_processor,
            self.embedding_manager,
            self.redis_store
        )
        
        logger.info("Server initialization complete with non-blocking indexing capabilities")

    async def shutdown(self) -> None:
        """Shutdown all server components gracefully.

        This method ensures all active components are properly closed and cleaned up,
        preventing resource leaks and ensuring data integrity. Components are shutdown
        in reverse dependency order to avoid errors.

        The shutdown process:
        1. Stop file watcher to prevent new indexing requests
        2. Close Redis connections and connection pools
        3. Clean up any remaining background tasks

        Raises:
            Exception: If critical shutdown operations fail. Non-critical failures
                are logged but don't prevent shutdown completion.

        Example:
            >>> server = EOLRAGContextServer()
            >>> await server.initialize()
            >>> # ... use server ...
            >>> await server.shutdown()
            Server shutdown complete

        """
        logger.info("Shutting down server")

        if self.file_watcher:
            await self.file_watcher.stop()

        if self.redis_store:
            await self.redis_store.close()

        logger.info("Server shutdown complete")

    def _setup_resources(self) -> None:
        """Configure MCP resources for dynamic context retrieval.

        Sets up MCP resource endpoints that provide dynamic access to context data.
        Resources are URI-based and allow clients to fetch specific information
        without requiring tool calls.

        Configured resources:
        - context://query/{query}: Get optimized context for a specific query
        - context://sources: List all indexed sources with metadata
        - context://stats: Get comprehensive server statistics
        - context://knowledge-graph/stats: Get knowledge graph statistics

        Note:
            This method should only be called during server initialization.
            Resources are automatically registered with the MCP server instance.

        """

        @self.mcp.resource("context://query/{query}")
        async def get_context_for_query(query: str) -> dict[str, Any]:
            """Get optimized context for a query using semantic cache and hierarchical
            search.

            Args:
                query: The search query string.

            Returns:
                Dictionary containing:
                - query: The original query string
                - context: The retrieved context text
                - cached: Whether result came from cache
                - source: Data source (semantic_cache or hierarchical_search)
                - results: Number of results found (if not cached)

            """
            # Check cache first
            cached = await self.semantic_cache.get(query)
            if cached:
                return {
                    "query": query,
                    "context": cached,
                    "cached": True,
                    "source": "semantic_cache",
                }

            # Perform hierarchical search
            query_embedding = await self.embedding_manager.get_embedding(query)
            results = await self.redis_store.hierarchical_search(
                query_embedding, max_chunks=self.config.context.default_top_k
            )

            # Format context
            context_parts = []
            for result in results:
                if result["score"] >= self.config.context.min_relevance_score:
                    context_parts.append(result["content"])

            context = "\n\n".join(context_parts)

            # Cache the result
            await self.semantic_cache.set(query, context)

            return {
                "query": query,
                "context": context,
                "cached": False,
                "results": len(results),
                "source": "hierarchical_search",
            }

        @self.mcp.resource("context://sources")
        async def list_indexed_sources() -> list[dict[str, Any]]:
            """List all indexed sources with their metadata and statistics.

            Returns:
                List of dictionaries, each containing:
                - source_id: Unique identifier for the source
                - path: Filesystem path to the indexed directory/file
                - indexed_at: ISO timestamp of when indexing completed
                - file_count: Number of files in this source
                - total_chunks: Total chunks created from this source
                - metadata: Additional source-specific metadata

            """
            # Ensure components are initialized
            if self.indexer is None:
                await self.initialize()
                
            sources = await self.indexer.list_sources()
            return [
                {
                    "source_id": source.source_id,
                    "path": str(source.path),
                    "indexed_at": datetime.fromtimestamp(source.indexed_at).isoformat(),
                    "file_count": source.file_count,
                    "total_chunks": source.total_chunks,
                    "metadata": source.metadata,
                }
                for source in sources
            ]

        @self.mcp.resource("context://stats")
        async def get_statistics() -> dict[str, Any]:
            """Get comprehensive server statistics from all components.

            Returns:
                Dictionary containing statistics from:
                - indexer: Document indexing metrics
                - cache: Semantic cache performance metrics
                - embeddings: Embedding generation and cache statistics
                - watcher: File watching activity metrics
                - knowledge_graph: Graph construction and query statistics

            """
            # Ensure components are initialized
            if self.indexer is None:
                await self.initialize()
                
            return {
                "indexer": self.indexer.get_stats(),
                "cache": self.semantic_cache.get_stats(),
                "embeddings": self.embedding_manager.get_cache_stats(),
                "watcher": self.file_watcher.get_stats(),
                "knowledge_graph": self.knowledge_graph.get_graph_stats(),
            }

        @self.mcp.resource("context://knowledge-graph/stats")
        async def get_knowledge_graph_stats() -> dict[str, Any]:
            """Get detailed knowledge graph statistics and metrics.

            Returns:
                Dictionary containing:
                - nodes: Total number of entities in the graph
                - edges: Total number of relationships
                - node_types: Distribution of entity types
                - relationship_types: Distribution of relationship types
                - centrality_metrics: Key centrality measurements
                - community_stats: Community detection results

            """
            return self.knowledge_graph.get_graph_stats()

    def _setup_tools(self) -> None:
        """Configure MCP tools for interactive operations.

        Sets up MCP tool endpoints that allow clients to perform operations
        like indexing, searching, and managing the RAG system. Each tool
        accepts structured parameters and returns formatted results.

        Configured tools:
        - index_directory: Index documents in a directory
        - search_context: Search for relevant context
        - query_knowledge_graph: Query entity relationships
        - optimize_context: Optimize context for LLM consumption
        - watch_directory: Start file watching
        - unwatch_directory: Stop file watching
        - clear_cache: Clear all caches
        - remove_source: Remove indexed source

        Note:
            This method should only be called during server initialization.
            Tools are automatically registered with the MCP server instance.

        """

        @self.mcp.tool()
        async def start_indexing(
            path: str,
            recursive: bool = True,
            force_reindex: bool = False,
            watch: bool = False,
            max_workers: int = 16,
            batch_size: int = 32,
            enable_streaming: bool = True,
            ctx: Context = None
        ) -> dict[str, Any]:
            """Start asynchronous indexing and return task ID immediately.
            
            This tool starts indexing in the background and returns immediately
            with a task ID that can be used to check progress. The AI agent is
            never blocked waiting for indexing to complete.
            
            Args:
                path: Directory path to index
                recursive: Index subdirectories (default: True)
                force_reindex: Force reindex of unchanged files (default: False)
                watch: Watch for changes after indexing (default: False)
                max_workers: Maximum concurrent workers (default: 16)
                batch_size: Batch size for processing (default: 32)
                enable_streaming: Enable streaming for large files (default: True)
                ctx: MCP context for the request
                
            Returns:
                Dictionary containing:
                - task_id: Unique identifier for tracking this indexing operation
                - status: Initial task status (pending)
                - path: Directory path being indexed
                - message: Human-readable status message
                - estimated_files: Rough estimate of files to be processed
            """
            # Ensure components are initialized
            if not self.task_manager or not self.parallel_indexer:
                await self.initialize()
            
            path_obj = Path(path).resolve()
            if not path_obj.exists() or not path_obj.is_dir():
                raise ValueError(f"Directory does not exist: {path_obj}")
            
            # Create parallel config
            parallel_config = ParallelIndexingConfig(
                max_document_workers=max_workers,
                max_embedding_workers=max_workers // 2,
                max_redis_workers=max_workers // 4,
                batch_size=batch_size,
                enable_streaming=enable_streaming
            )
            
            # Start indexing task
            task_id = await self.task_manager.start_indexing_task(
                path_obj,
                self.parallel_indexer,
                recursive=recursive,
                force_reindex=force_reindex,
                parallel_config=parallel_config
            )
            
            # Quick file count estimate
            estimated_files = self._estimate_file_count(path_obj, recursive)
            
            # Start watching if requested (this is immediate and non-blocking)
            if watch:
                await self.file_watcher.watch(
                    path_obj,
                    recursive=recursive,
                    file_patterns=None,
                )
            
            return {
                "task_id": task_id,
                "status": "pending",
                "path": str(path_obj),
                "message": f"Indexing started for {path_obj} with {max_workers} workers",
                "estimated_files": estimated_files,
                "watching": watch,
                "parallel_config": {
                    "max_workers": max_workers,
                    "batch_size": batch_size,
                    "streaming_enabled": enable_streaming
                }
            }

        @self.mcp.tool()
        async def get_indexing_status(task_id: str, ctx: Context) -> dict[str, Any]:
            """Get current status and progress of an indexing task.
            
            Args:
                task_id: Task ID to check status for
                ctx: MCP context for the request
                
            Returns:
                Dictionary containing detailed task status and progress information
            """
            # Ensure components are initialized
            if not self.task_manager:
                await self.initialize()
            
            task_info = await self.task_manager.get_task_status(task_id)
            
            if not task_info:
                return {
                    "error": "Task not found",
                    "task_id": task_id,
                    "message": "No indexing task found with the specified ID"
                }
            
            result = {
                "task_id": task_info.task_id,
                "status": task_info.status.value,
                "path": task_info.folder_path,
                "progress": {
                    "completed_files": task_info.completed_files,
                    "total_files": task_info.total_files,
                    "percentage": task_info.progress_percentage,
                    "files_per_second": task_info.files_per_second
                },
                "timing": {
                    "created_at": datetime.fromtimestamp(task_info.created_at).isoformat(),
                    "elapsed_time": task_info.elapsed_time
                },
                "results": {
                    "total_chunks": task_info.total_chunks,
                    "indexed_files": task_info.indexed_files,
                    "failed_files": task_info.failed_files
                }
            }
            
            # Add completion estimate for running tasks
            if task_info.status == TaskStatus.RUNNING and task_info.estimated_completion_time:
                result["timing"]["estimated_completion_seconds"] = task_info.estimated_completion_time
                result["timing"]["estimated_completion"] = datetime.fromtimestamp(
                    task_info.created_at + task_info.estimated_completion_time
                ).isoformat()
            
            # Add current file being processed
            if task_info.current_file:
                result["current_file"] = task_info.current_file
            
            # Add error information for failed tasks
            if task_info.status == TaskStatus.FAILED:
                result["error"] = {
                    "message": task_info.error_message,
                    "details": task_info.errors[:10] if task_info.errors else []  # Limit error details
                }
            
            # Add result information for completed tasks
            if task_info.status == TaskStatus.COMPLETED and task_info.result:
                result["indexing_result"] = {
                    "source_id": task_info.result.get("source_id"),
                    "indexed_at": datetime.fromtimestamp(
                        task_info.result.get("indexed_at", task_info.completed_at)
                    ).isoformat(),
                    "performance": task_info.result.get("metadata", {}).get("performance", {})
                }
            
            return result

        @self.mcp.tool()
        async def list_indexing_tasks(status_filter: str = None, limit: int = 50, ctx: Context = None) -> dict[str, Any]:
            """List indexing tasks with optional status filtering.
            
            Args:
                status_filter: Filter by status (pending, running, completed, failed, cancelled)
                limit: Maximum tasks to return (default: 50)
                ctx: MCP context for the request
                
            Returns:
                Dictionary containing list of tasks and summary statistics
            """
            # Ensure components are initialized
            if not self.task_manager:
                await self.initialize()
            
            # Convert status filter
            status_filter_enum = None
            if status_filter:
                try:
                    status_filter_enum = TaskStatus(status_filter.lower())
                except ValueError:
                    return {
                        "error": f"Invalid status filter: {status_filter}",
                        "valid_statuses": [s.value for s in TaskStatus]
                    }
            
            tasks = await self.task_manager.list_tasks(status_filter_enum, limit)
            
            # Convert to response format
            task_list = []
            for task_info in tasks:
                task_summary = {
                    "task_id": task_info.task_id,
                    "status": task_info.status.value,
                    "path": task_info.folder_path,
                    "created_at": datetime.fromtimestamp(task_info.created_at).isoformat(),
                    "progress_percentage": task_info.progress_percentage,
                    "completed_files": task_info.completed_files,
                    "total_files": task_info.total_files,
                    "elapsed_time": task_info.elapsed_time
                }
                
                if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    task_summary["completed_at"] = datetime.fromtimestamp(
                        task_info.completed_at
                    ).isoformat()
                
                task_list.append(task_summary)
            
            # Calculate summary statistics
            status_counts = {}
            for task in tasks:
                status = task.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "tasks": task_list,
                "total_tasks": len(task_list),
                "status_summary": status_counts,
                "filter_applied": status_filter,
                "limit_applied": limit
            }

        @self.mcp.tool()
        async def cancel_indexing_task(task_id: str, ctx: Context = None) -> dict[str, Any]:
            """Cancel a running indexing task.
            
            Args:
                task_id: Task ID to cancel
                ctx: MCP context for the request
                
            Returns:
                Dictionary containing cancellation result
            """
            # Ensure components are initialized
            if not self.task_manager:
                await self.initialize()
            
            success = await self.task_manager.cancel_task(task_id)
            
            return {
                "task_id": task_id,
                "cancelled": success,
                "message": (
                    "Task cancelled successfully" if success 
                    else "Task not found or not cancellable"
                )
            }

        @self.mcp.tool()
        async def cleanup_old_indexing_tasks(ctx: Context) -> dict[str, Any]:
            """Clean up old completed/failed indexing tasks.
            
            Removes old tasks from memory and Redis to free up resources.
            Tasks older than 24 hours are automatically cleaned up.
            
            Args:
                ctx: MCP context for the request
                
            Returns:
                Dictionary containing cleanup results
            """
            # Ensure components are initialized
            if not self.task_manager:
                await self.initialize()
            
            cleaned_count = await self.task_manager.cleanup_old_tasks()
            
            return {
                "cleaned_tasks": cleaned_count,
                "message": f"Cleaned up {cleaned_count} old indexing tasks"
            }

        @self.mcp.tool()
        async def search_context(
            query: str,
            max_results: int = 10,
            min_relevance: float = 0.7,
            hierarchy_level: int = None,
            source_filter: str = None,
            ctx: Context = None
        ) -> list[dict[str, Any]]:
            """Search for relevant context using vector similarity.

            Performs vector similarity search against indexed documents, supporting
            both hierarchical search across all levels and targeted search at
            specific hierarchy levels.

            Args:
                query: Search query string
                max_results: Maximum number of results to return (default: 10)
                min_relevance: Minimum similarity score threshold (default: 0.7)
                hierarchy_level: Optional specific level (1=concept, 2=section, 3=chunk)
                source_filter: Optional source ID to filter results
                ctx: MCP context for the request

            Returns:
                List of dictionaries, each containing:
                - id: Document/chunk identifier
                - score: Similarity score (0.0 to 1.0)
                - content: The retrieved content text
                - metadata: Document metadata including source, hierarchy info

            """
            # Get query embedding
            query_embedding = await self.embedding_manager.get_embedding(query)

            # Perform search
            if hierarchy_level:
                results = await self.redis_store.vector_search(
                    query_embedding,
                    hierarchy_level=hierarchy_level,
                    k=max_results,
                    filters=(
                        {"source_id": source_filter} if source_filter else None
                    ),
                )

                return [
                    {
                        "id": doc_id,
                        "score": float(score),
                        "content": data["content"],
                        "metadata": data["metadata"],
                    }
                    for doc_id, score, data in results
                    if score >= min_relevance
                ]
            else:
                # Hierarchical search
                results = await self.redis_store.hierarchical_search(
                    query_embedding, max_chunks=max_results
                )

                return [result for result in results if result["score"] >= min_relevance]

        @self.mcp.tool()
        async def query_knowledge_graph(
            query: str,
            max_depth: int = 2,
            max_entities: int = 20,
            ctx: Context = None
        ) -> dict[str, Any]:
            """Query the knowledge graph for entity relationships.

            Searches the knowledge graph for entities matching the query and returns
            a subgraph containing related entities and their relationships within
            the specified depth limit.

            Args:
                request: QueryKnowledgeGraphRequest containing:
                    - query: Entity name or description to search for
                    - max_depth: Maximum relationship traversal depth
                    - max_entities: Maximum number of entities to return
                ctx: MCP context for the request

            Returns:
                Dictionary containing:
                - query: The original query string
                - entities: List of matching entities with properties
                - relationships: List of relationships between entities
                - central_entities: Most connected entities in the subgraph
                - metadata: Additional subgraph statistics

            """
            subgraph = await self.knowledge_graph.query_subgraph(
                query,
                max_depth=max_depth,
                max_entities=max_entities,
            )

            return {
                "query": query,
                "entities": [
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.type.value,
                        "content": entity.content[:200],
                        "properties": entity.properties,
                    }
                    for entity in subgraph.entities
                ],
                "relationships": [
                    {
                        "source": rel.source_id,
                        "target": rel.target_id,
                        "type": rel.type.value,
                        "weight": rel.weight,
                    }
                    for rel in subgraph.relationships
                ],
                "central_entities": subgraph.central_entities,
                "metadata": subgraph.metadata,
            }

        @self.mcp.tool()
        async def optimize_context(
            query: str,
            current_context: str = None,
            max_tokens: int = 32000,
            strategy: str = "hierarchical",
            ctx: Context = None
        ) -> dict[str, Any]:
            """Optimize context for LLM consumption following best practices.

            Retrieves relevant context and formats it optimally for LLM processing,
            including hierarchical organization, token management, and structured
            presentation following 2024 LLM context best practices.

            Args:
                request: OptimizeContextRequest containing:
                    - query: User query for context retrieval
                    - current_context: Optional existing context to optimize
                    - max_tokens: Maximum token limit for output context
                    - strategy: Context organization strategy
                ctx: MCP context for the request

            Returns:
                Dictionary containing:
                - query: The original query string
                - optimized_context: Formatted context text ready for LLM
                - total_results: Number of source results used
                - strategy: Strategy used for optimization
                - estimated_tokens: Approximate token count of output

            """
            # Get relevant context
            query_embedding = await self.embedding_manager.get_embedding(query)

            # Hierarchical retrieval
            results = await self.redis_store.hierarchical_search(
                query_embedding, max_chunks=20, strategy=strategy
            )

            # Build optimized context following best practices
            context_parts = []

            # 1. System instructions (if needed)
            context_parts.append("## Retrieved Context\n")

            # 2. High-level concepts first
            concepts = [r for r in results if r.get("hierarchy", {}).get("concept")]
            if concepts:
                context_parts.append("### Key Concepts:")
                for concept in concepts[:2]:
                    context_parts.append(f"- {concept['content'][:200]}")

            # 3. Relevant sections
            sections = [r for r in results if r.get("hierarchy", {}).get("section")]
            if sections:
                context_parts.append("\n### Relevant Information:")
                for section in sections[:5]:
                    context_parts.append(section["content"])

            # 4. Specific details
            chunks = [r for r in results if r.get("hierarchy", {}).get("chunk")]
            if chunks:
                context_parts.append("\n### Specific Details:")
                for chunk in chunks[:10]:
                    context_parts.append(f"- {chunk['content'][:300]}")

            optimized_context = "\n\n".join(context_parts)

            # Trim to token limit
            # Simple approximation: ~4 chars per token
            max_chars = max_tokens * 4
            if len(optimized_context) > max_chars:
                optimized_context = optimized_context[:max_chars] + "\n\n[Context truncated]"

            return {
                "query": query,
                "optimized_context": optimized_context,
                "total_results": len(results),
                "strategy": strategy,
                "estimated_tokens": len(optimized_context) // 4,
            }

        @self.mcp.tool()
        async def watch_directory(
            path: str,
            recursive: bool = True,
            file_patterns: list[str] = None,
            ctx: Context = None
        ) -> dict[str, Any]:
            """Start watching a directory for file changes.

            Begins monitoring the specified directory for file changes, automatically
            reindexing modified files and updating the knowledge graph as needed.

            Args:
                path: Directory path to watch
                recursive: Whether to watch subdirectories (default: True)
                file_patterns: Optional glob patterns for file filtering
                ctx: MCP context for the request

            Returns:
                Dictionary containing:
                - source_id: Unique identifier for the watch session
                - path: Absolute path being watched
                - recursive: Whether subdirectories are included
                - patterns: File patterns being monitored
                - status: Current watching status

            """
            path_obj = Path(path)

            source_id = await self.file_watcher.watch(
                path_obj, recursive=recursive, file_patterns=file_patterns
            )

            return {
                "source_id": source_id,
                "path": str(path_obj),
                "recursive": recursive,
                "patterns": file_patterns,
                "status": "watching",
            }

        @self.mcp.tool()
        async def unwatch_directory(source_id: str, ctx: Context) -> dict[str, Any]:
            """Stop watching a directory for changes.

            Stops the file watcher for the specified source, preventing further
            automatic reindexing of changes in that directory.

            Args:
                source_id: Unique identifier of the watch session to stop
                ctx: MCP context for the request

            Returns:
                Dictionary containing:
                - source_id: The watch session identifier
                - unwatched: Whether the operation was successful
                - status: Final status (stopped or not_found)

            """
            success = await self.file_watcher.unwatch(source_id)

            return {
                "source_id": source_id,
                "unwatched": success,
                "status": "stopped" if success else "not_found",
            }

        @self.mcp.tool()
        async def clear_cache(ctx: Context) -> dict[str, Any]:
            """Clear all caches to force fresh data retrieval.

            Clears both semantic cache and embedding cache, forcing all subsequent
            queries to regenerate embeddings and retrieve fresh context data.

            Args:
                ctx: MCP context for the request

            Returns:
                Dictionary containing:
                - semantic_cache: Clearance status
                - embedding_cache: Clearance status
                - timestamp: ISO timestamp of operation

            """
            # Clear semantic cache
            await self.semantic_cache.clear()

            # Clear embedding cache
            await self.embedding_manager.clear_cache()

            return {
                "semantic_cache": "cleared",
                "embedding_cache": "cleared",
                "timestamp": datetime.now().isoformat(),
            }

        @self.mcp.tool()
        async def remove_source(source_id: str, ctx: Context) -> dict[str, Any]:
            """Remove an indexed source and all its data.

            Completely removes all indexed data for the specified source,
            including documents, embeddings, knowledge graph entities,
            and stops any active file watching.

            Args:
                source_id: Unique identifier of the source to remove
                ctx: MCP context for the request

            Returns:
                Dictionary containing:
                - source_id: The source identifier
                - removed: Whether the operation was successful
                - status: Final status (removed or not_found)

            """
            # Stop watching if active
            await self.file_watcher.unwatch(source_id)

            # Remove from index
            success = await self.indexer.remove_source(source_id)

            return {
                "source_id": source_id,
                "removed": success,
                "status": "removed" if success else "not_found",
            }

    def _setup_prompts(self) -> None:
        """Configure MCP prompts for structured LLM interactions.

        Sets up prompt templates that help LLMs interact more effectively
        with the RAG system by providing structured formats for queries,
        context synthesis, and knowledge exploration.

        Configured prompts:
        - structured_query: Transform user queries for optimal RAG retrieval
        - context_synthesis: Synthesize multiple context sections effectively
        - knowledge_exploration: Explore knowledge graphs for insights

        Note:
            This method should only be called during server initialization.
            Prompts are automatically registered with the MCP server instance.

        """

        @self.mcp.prompt("structured_query")
        async def structured_query_prompt() -> str:
            """Generate a structured query format for optimal RAG retrieval.

            Provides a template for transforming user queries into structured
            formats that improve RAG retrieval effectiveness by identifying
            intent, entities, required depth, and output format.

            Returns:
                Structured query template with examples.

            """
            return """Transform the user query into a structured format for optimal RAG retrieval:

1. Main Intent: [What is the user trying to achieve?]
2. Key Entities: [Important nouns, concepts, or terms]
3. Context Level: [concept/section/detail - what depth is needed?]
4. Required Depth: [shallow/medium/deep - how comprehensive?]
5. Output Format: [explanation/code/list/comparison - what format?]

Example:
User: "How does the authentication system work?"

1. Main Intent: Understand authentication system architecture
2. Key Entities: authentication, system, login, security
3. Context Level: section
4. Required Depth: medium
5. Output Format: explanation"""

        @self.mcp.prompt("context_synthesis")
        async def context_synthesis_prompt() -> str:
            """Synthesize multiple context sections into coherent responses.

            Provides guidelines for effectively combining multiple retrieved
            context sections, resolving contradictions, and organizing
            information hierarchically for optimal LLM understanding.

            Returns:
                Context synthesis template with formatting guidelines.

            """
            return """Given the retrieved context sections, synthesize them effectively:

1. Identify Common Themes: Find recurring concepts across sections
2. Resolve Contradictions: If information conflicts, note the discrepancy
3. Hierarchical Organization: Structure from general to specific
4. Highlight Relevance: Emphasize parts most relevant to the query
5. Summarize if Needed: Condense if exceeding token limits

Output Format:
## Overview
[High-level summary]

## Key Points
- [Main point 1]
- [Main point 2]

## Detailed Information
[Relevant details organized by topic]

## Related Concepts
[Connected ideas and references]"""

        @self.mcp.prompt("knowledge_exploration")
        async def knowledge_exploration_prompt() -> str:
            """Explore knowledge graphs to discover insights and patterns.

            Provides structured approach for analyzing knowledge graphs,
            including entity analysis, relationship patterns, community
            detection, and path analysis to uncover hidden insights.

            Returns:
                Knowledge exploration template with analysis framework.

            """
            return """Explore the knowledge graph to discover insights:

1. Entity Analysis:
   - What are the most connected entities (hubs)?
   - What types of entities are present?
   - Are there isolated entities?

2. Relationship Patterns:
   - What are the most common relationship types?
   - Are there circular dependencies?
   - What are the strongest connections?

3. Community Detection:
   - Are there distinct clusters or communities?
   - What characterizes each community?
   - How are communities connected?

4. Path Analysis:
   - What are the shortest paths between key entities?
   - Are there critical paths or bottlenecks?
   - What are alternative routes?

5. Recommendations:
   - What entities might be related but aren't connected?
   - What documentation gaps exist?
   - What refactoring opportunities are evident?"""

    def _estimate_file_count(self, path: Path, recursive: bool) -> int:
        """Quick estimate of files to be processed."""
        try:
            if recursive:
                # Quick estimate by sampling
                count = 0
                for _ in path.rglob("*"):
                    count += 1
                    if count > 10000:  # Cap the estimate
                        return 10000
                return count
            else:
                return len([f for f in path.glob("*") if f.is_file()])
        except Exception:
            return 0  # Return 0 if we can't estimate

    # API compatibility methods for tests and external usage

    async def index_directory(self, path: str, **kwargs) -> dict[str, Any]:
        """Start non-blocking directory indexing (API compatibility method).

        DEPRECATED: This method is provided for backward compatibility only.
        New code should use the start_indexing MCP tool directly for better control.
        
        This method starts indexing in the background and returns a task ID.
        Use get_indexing_status() to check progress.

        Args:
            path: Directory path to index
            **kwargs: Additional indexing options:
                recursive (bool): Process subdirectories. Defaults to True.
                force_reindex (bool): Reindex even if unchanged. Defaults to False.
                max_workers (int): Maximum concurrent workers. Defaults to 16.

        Returns:
            Dictionary containing:
            - status: "started" (instead of "success" to indicate async nature)
            - task_id: Unique identifier for tracking progress
            - message: Information about task start
            - path: Directory path being indexed

        """
        # Ensure components are initialized
        if not self.task_manager or not self.parallel_indexer:
            await self.initialize()

        # Extract parameters
        recursive = kwargs.get("recursive", True)
        force_reindex = kwargs.get("force_reindex", False)
        max_workers = kwargs.get("max_workers", 16)

        try:
            # Create parallel config
            parallel_config = ParallelIndexingConfig(
                max_document_workers=max_workers,
                max_embedding_workers=max_workers // 2,
                max_redis_workers=max_workers // 4,
                batch_size=32,
                enable_streaming=True
            )
            
            # Start indexing task
            task_id = await self.task_manager.start_indexing_task(
                Path(path),
                self.parallel_indexer,
                recursive=recursive,
                force_reindex=force_reindex,
                parallel_config=parallel_config
            )

            return {
                "status": "started",  # Changed from "success" to indicate async nature
                "task_id": task_id,
                "message": f"Non-blocking indexing started for {path}. Use get_indexing_status(task_id) to track progress.",
                "path": path,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def index_file(self, path: str, **kwargs) -> dict[str, Any]:
        """Index a single file with dict return for API compatibility.

        Processes a single file, extracting content, creating chunks, generating
        embeddings, and storing in Redis. This method provides backward
        compatibility with previous API versions.

        Args:
            path: Filesystem path to the file to index
            **kwargs: Additional indexing options (currently unused)

        Returns:
            Dictionary containing:
            - status: "success" or "error"
            - source_id: Unique identifier for this indexed file
            - chunks: Number of chunks created from this file
            - total_chunks: Same as chunks (for consistency)
            - files: Number of files processed (always 1)
            - errors: List of any errors encountered
            - message: Error message if status is "error"

        """
        if not self.indexer:
            return {"status": "error", "message": "Indexer not initialized"}

        try:
            result = await self.indexer.index_file(path)

            # Convert IndexResult to dict for compatibility
            return {
                "status": "success",
                "source_id": result.source_id,
                "chunks": result.chunks,
                "total_chunks": result.total_chunks,
                "files": result.files,
                "errors": result.errors if result.errors else [],
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def watch_directory(self, path: str, **kwargs) -> dict[str, Any]:
        """Watch a directory for changes with dict return for API compatibility.

        Starts monitoring the specified directory for file changes, automatically
        triggering reindexing when files are modified, added, or deleted.

        Args:
            path: Directory path to watch
            **kwargs: Additional watching options (currently unused)

        Returns:
            Dictionary containing:
            - status: "success" or "error"
            - path: The directory path being watched
            - message: Success or error message

        """
        if not self.file_watcher:
            return {"status": "error", "message": "File watcher not initialized"}

        try:
            # Start watching the directory
            await self.file_watcher.start_watching(Path(path))
            return {
                "status": "success",
                "path": path,
                "message": f"Now watching {path} for changes",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def run(self) -> None:
        """Run the MCP server with full initialization and cleanup.

        Initializes all server components, starts the MCP server, and ensures
        proper shutdown when the server stops. This is the main entry point
        for running the server as a standalone application.

        Raises:
            Exception: If server initialization or startup fails.

        Note:
            This method runs indefinitely until the server is stopped.
            Use Ctrl+C or send SIGTERM to gracefully shutdown.

        """
        await self.initialize()

        try:
            # Start MCP server
            await self.mcp.run()
        finally:
            await self.shutdown()


async def main():
    """Main entry point for the EOL RAG Context MCP server.

    Configures logging, loads configuration from command line arguments,
    creates the server instance, and runs it with proper error handling.

    Command line usage:
        python -m eol.rag_context.server [config.json]

    Args:
        sys.argv[1] (optional): Path to configuration file. If not provided,
            uses default configuration.

    Example:
        Run with default configuration:
        $ python -m eol.rag_context.server

        Run with custom configuration:
        $ python -m eol.rag_context.server /path/to/config.json

    """
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    config = RAGConfig.from_file(config_path) if config_path else RAGConfig()

    # Create and run server
    server = EOLRAGContextServer(config)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
