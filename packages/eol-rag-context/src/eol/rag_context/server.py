"""
EOL RAG Context MCP Server.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastmcp import FastMCP, Context
from pydantic import BaseModel, Field

from .config import RAGConfig
from .redis_client import RedisVectorStore
from .embeddings import EmbeddingManager
from .document_processor import DocumentProcessor
from .indexer import DocumentIndexer
from .semantic_cache import SemanticCache
from .knowledge_graph import KnowledgeGraphBuilder
from .file_watcher import FileWatcher

logger = logging.getLogger(__name__)


# Pydantic models for MCP tools
class IndexDirectoryRequest(BaseModel):
    """Request to index a directory."""
    path: str = Field(description="Directory path to index")
    recursive: bool = Field(default=True, description="Index subdirectories")
    file_patterns: Optional[List[str]] = Field(default=None, description="File patterns to index")
    watch: bool = Field(default=False, description="Watch for changes after indexing")


class SearchContextRequest(BaseModel):
    """Request to search for context."""
    query: str = Field(description="Search query")
    max_results: int = Field(default=10, description="Maximum results to return")
    min_relevance: float = Field(default=0.7, description="Minimum relevance score")
    hierarchy_level: Optional[int] = Field(default=None, description="Search at specific hierarchy level (1=concept, 2=section, 3=chunk)")
    source_filter: Optional[str] = Field(default=None, description="Filter by source ID")


class QueryKnowledgeGraphRequest(BaseModel):
    """Request to query knowledge graph."""
    query: str = Field(description="Query for knowledge graph")
    max_depth: int = Field(default=2, description="Maximum traversal depth")
    max_entities: int = Field(default=20, description="Maximum entities to return")


class OptimizeContextRequest(BaseModel):
    """Request to optimize context for LLM."""
    query: str = Field(description="User query")
    current_context: Optional[str] = Field(default=None, description="Current context to optimize")
    max_tokens: int = Field(default=32000, description="Maximum context tokens")
    strategy: str = Field(default="hierarchical", description="Context strategy (hierarchical, flat, semantic)")


class WatchDirectoryRequest(BaseModel):
    """Request to watch a directory for changes."""
    path: str = Field(description="Directory path to watch")
    recursive: bool = Field(default=True, description="Watch subdirectories")
    file_patterns: Optional[List[str]] = Field(default=None, description="File patterns to watch")


class EOLRAGContextServer:
    """MCP server for intelligent RAG-based context management."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.mcp = FastMCP(
            name=self.config.server_name,
            version=self.config.server_version
        )
        
        # Core components
        self.redis_store: Optional[RedisVectorStore] = None
        self.embedding_manager: Optional[EmbeddingManager] = None
        self.document_processor: Optional[DocumentProcessor] = None
        self.indexer: Optional[DocumentIndexer] = None
        self.semantic_cache: Optional[SemanticCache] = None
        self.knowledge_graph: Optional[KnowledgeGraphBuilder] = None
        self.file_watcher: Optional[FileWatcher] = None
        
        # Setup MCP handlers
        self._setup_resources()
        self._setup_tools()
        self._setup_prompts()
    
    async def initialize(self) -> None:
        """Initialize server components."""
        logger.info(f"Initializing {self.config.server_name} v{self.config.server_version}")
        
        # Initialize Redis
        self.redis_store = RedisVectorStore(self.config.redis, self.config.index)
        await self.redis_store.connect_async()
        
        # Initialize embeddings
        self.embedding_manager = EmbeddingManager(
            self.config.embedding,
            self.redis_store.async_redis
        )
        
        # Create indexes
        self.redis_store.create_hierarchical_indexes(self.config.embedding.dimension)
        
        # Initialize processors
        self.document_processor = DocumentProcessor(
            self.config.document,
            self.config.chunking
        )
        
        self.indexer = DocumentIndexer(
            self.config,
            self.document_processor,
            self.embedding_manager,
            self.redis_store
        )
        
        # Initialize semantic cache
        self.semantic_cache = SemanticCache(
            self.config.cache,
            self.embedding_manager,
            self.redis_store
        )
        await self.semantic_cache.initialize()
        
        # Initialize knowledge graph
        self.knowledge_graph = KnowledgeGraphBuilder(
            self.redis_store,
            self.embedding_manager
        )
        
        # Initialize file watcher
        self.file_watcher = FileWatcher(
            self.indexer,
            self.knowledge_graph,
            debounce_seconds=2.0
        )
        await self.file_watcher.start()
        
        logger.info("Server initialization complete")
    
    async def shutdown(self) -> None:
        """Shutdown server components."""
        logger.info("Shutting down server")
        
        if self.file_watcher:
            await self.file_watcher.stop()
        
        if self.redis_store:
            await self.redis_store.close()
        
        logger.info("Server shutdown complete")
    
    def _setup_resources(self) -> None:
        """Setup MCP resources."""
        
        @self.mcp.resource("context://query/{query}")
        async def get_context_for_query(query: str) -> Dict[str, Any]:
            """Get optimized context for a query."""
            # Check cache first
            cached = await self.semantic_cache.get(query)
            if cached:
                return {
                    "query": query,
                    "context": cached,
                    "cached": True,
                    "source": "semantic_cache"
                }
            
            # Perform hierarchical search
            query_embedding = await self.embedding_manager.get_embedding(query)
            results = await self.redis_store.hierarchical_search(
                query_embedding,
                max_chunks=self.config.context.default_top_k
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
                "source": "hierarchical_search"
            }
        
        @self.mcp.resource("context://sources")
        async def list_indexed_sources() -> List[Dict[str, Any]]:
            """List all indexed sources."""
            sources = await self.indexer.list_sources()
            return [
                {
                    "source_id": source.source_id,
                    "path": str(source.path),
                    "indexed_at": datetime.fromtimestamp(source.indexed_at).isoformat(),
                    "file_count": source.file_count,
                    "total_chunks": source.total_chunks,
                    "metadata": source.metadata
                }
                for source in sources
            ]
        
        @self.mcp.resource("context://stats")
        async def get_statistics() -> Dict[str, Any]:
            """Get server statistics."""
            return {
                "indexer": self.indexer.get_stats(),
                "cache": self.semantic_cache.get_stats(),
                "embeddings": self.embedding_manager.get_cache_stats(),
                "watcher": self.file_watcher.get_stats(),
                "knowledge_graph": self.knowledge_graph.get_graph_stats()
            }
        
        @self.mcp.resource("context://knowledge-graph/stats")
        async def get_knowledge_graph_stats() -> Dict[str, Any]:
            """Get knowledge graph statistics."""
            return self.knowledge_graph.get_graph_stats()
    
    def _setup_tools(self) -> None:
        """Setup MCP tools."""
        
        @self.mcp.tool()
        async def index_directory(
            request: IndexDirectoryRequest,
            ctx: Context
        ) -> Dict[str, Any]:
            """Index a directory of documents."""
            path = Path(request.path)
            
            # Index the directory
            result = await self.indexer.index_folder(
                path,
                recursive=request.recursive,
                force_reindex=False
            )
            
            # Build knowledge graph
            await self.knowledge_graph.build_from_documents(result.source_id)
            
            # Start watching if requested
            if request.watch:
                await self.file_watcher.watch(
                    path,
                    recursive=request.recursive,
                    file_patterns=request.file_patterns
                )
            
            return {
                "source_id": result.source_id,
                "path": str(result.path),
                "indexed_at": datetime.fromtimestamp(result.indexed_at).isoformat(),
                "file_count": result.file_count,
                "total_chunks": result.total_chunks,
                "watching": request.watch
            }
        
        @self.mcp.tool()
        async def search_context(
            request: SearchContextRequest,
            ctx: Context
        ) -> List[Dict[str, Any]]:
            """Search for relevant context."""
            # Get query embedding
            query_embedding = await self.embedding_manager.get_embedding(request.query)
            
            # Perform search
            if request.hierarchy_level:
                results = await self.redis_store.vector_search(
                    query_embedding,
                    hierarchy_level=request.hierarchy_level,
                    k=request.max_results,
                    filters={"source_id": request.source_filter} if request.source_filter else None
                )
                
                return [
                    {
                        "id": doc_id,
                        "score": float(score),
                        "content": data["content"],
                        "metadata": data["metadata"]
                    }
                    for doc_id, score, data in results
                    if score >= request.min_relevance
                ]
            else:
                # Hierarchical search
                results = await self.redis_store.hierarchical_search(
                    query_embedding,
                    max_chunks=request.max_results
                )
                
                return [
                    result for result in results
                    if result["score"] >= request.min_relevance
                ]
        
        @self.mcp.tool()
        async def query_knowledge_graph(
            request: QueryKnowledgeGraphRequest,
            ctx: Context
        ) -> Dict[str, Any]:
            """Query the knowledge graph."""
            subgraph = await self.knowledge_graph.query_subgraph(
                request.query,
                max_depth=request.max_depth,
                max_entities=request.max_entities
            )
            
            return {
                "query": request.query,
                "entities": [
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.type.value,
                        "content": entity.content[:200],
                        "properties": entity.properties
                    }
                    for entity in subgraph.entities
                ],
                "relationships": [
                    {
                        "source": rel.source_id,
                        "target": rel.target_id,
                        "type": rel.type.value,
                        "weight": rel.weight
                    }
                    for rel in subgraph.relationships
                ],
                "central_entities": subgraph.central_entities,
                "metadata": subgraph.metadata
            }
        
        @self.mcp.tool()
        async def optimize_context(
            request: OptimizeContextRequest,
            ctx: Context
        ) -> Dict[str, Any]:
            """Optimize context for LLM consumption."""
            # Get relevant context
            query_embedding = await self.embedding_manager.get_embedding(request.query)
            
            # Hierarchical retrieval
            results = await self.redis_store.hierarchical_search(
                query_embedding,
                max_chunks=20,
                strategy=request.strategy
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
                    context_parts.append(section['content'])
            
            # 4. Specific details
            chunks = [r for r in results if r.get("hierarchy", {}).get("chunk")]
            if chunks:
                context_parts.append("\n### Specific Details:")
                for chunk in chunks[:10]:
                    context_parts.append(f"- {chunk['content'][:300]}")
            
            optimized_context = "\n\n".join(context_parts)
            
            # Trim to token limit
            # Simple approximation: ~4 chars per token
            max_chars = request.max_tokens * 4
            if len(optimized_context) > max_chars:
                optimized_context = optimized_context[:max_chars] + "\n\n[Context truncated]"
            
            return {
                "query": request.query,
                "optimized_context": optimized_context,
                "total_results": len(results),
                "strategy": request.strategy,
                "estimated_tokens": len(optimized_context) // 4
            }
        
        @self.mcp.tool()
        async def watch_directory(
            request: WatchDirectoryRequest,
            ctx: Context
        ) -> Dict[str, Any]:
            """Start watching a directory for changes."""
            path = Path(request.path)
            
            source_id = await self.file_watcher.watch(
                path,
                recursive=request.recursive,
                file_patterns=request.file_patterns
            )
            
            return {
                "source_id": source_id,
                "path": str(path),
                "recursive": request.recursive,
                "patterns": request.file_patterns,
                "status": "watching"
            }
        
        @self.mcp.tool()
        async def unwatch_directory(source_id: str, ctx: Context) -> Dict[str, Any]:
            """Stop watching a directory."""
            success = await self.file_watcher.unwatch(source_id)
            
            return {
                "source_id": source_id,
                "unwatched": success,
                "status": "stopped" if success else "not_found"
            }
        
        @self.mcp.tool()
        async def clear_cache(ctx: Context) -> Dict[str, Any]:
            """Clear all caches."""
            # Clear semantic cache
            await self.semantic_cache.clear()
            
            # Clear embedding cache
            await self.embedding_manager.clear_cache()
            
            return {
                "semantic_cache": "cleared",
                "embedding_cache": "cleared",
                "timestamp": datetime.now().isoformat()
            }
        
        @self.mcp.tool()
        async def remove_source(source_id: str, ctx: Context) -> Dict[str, Any]:
            """Remove an indexed source."""
            # Stop watching if active
            await self.file_watcher.unwatch(source_id)
            
            # Remove from index
            success = await self.indexer.remove_source(source_id)
            
            return {
                "source_id": source_id,
                "removed": success,
                "status": "removed" if success else "not_found"
            }
    
    def _setup_prompts(self) -> None:
        """Setup MCP prompts."""
        
        @self.mcp.prompt("structured_query")
        async def structured_query_prompt() -> str:
            """Generate a structured query for RAG retrieval."""
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
            """Synthesize multiple context sections."""
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
            """Explore knowledge graph for insights."""
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
    
    # API compatibility methods for tests and external usage
    
    async def index_directory(self, path: str, **kwargs) -> Dict[str, Any]:
        """Index a directory (alias for index_folder with dict return)."""
        if not self.indexer:
            return {"status": "error", "message": "Indexer not initialized"}
        
        # Extract supported parameters
        recursive = kwargs.get("recursive", True)
        force_reindex = kwargs.get("force_reindex", False)
        
        try:
            result = await self.indexer.index_folder(
                path,
                recursive=recursive,
                force_reindex=force_reindex
            )
            
            # Convert IndexedSource to dict for compatibility
            return {
                "status": "success",
                "source_id": result.source_id,
                "indexed_files": result.indexed_files,
                "total_chunks": result.total_chunks,
                "file_count": result.file_count,
                "path": str(result.path)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def index_file(self, path: str, **kwargs) -> Dict[str, Any]:
        """Index a single file with dict return for compatibility."""
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
                "errors": result.errors if result.errors else []
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def watch_directory(self, path: str, **kwargs) -> Dict[str, Any]:
        """Watch a directory for changes."""
        if not self.file_watcher:
            return {"status": "error", "message": "File watcher not initialized"}
        
        try:
            # Start watching the directory
            await self.file_watcher.start_watching(Path(path))
            return {
                "status": "success",
                "path": path,
                "message": f"Now watching {path} for changes"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def run(self) -> None:
        """Run the MCP server."""
        await self.initialize()
        
        try:
            # Start MCP server
            await self.mcp.run()
        finally:
            await self.shutdown()


async def main():
    """Main entry point for the server."""
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    config = RAGConfig.from_file(config_path) if config_path else RAGConfig()
    
    # Create and run server
    server = EOLRAGContextServer(config)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())