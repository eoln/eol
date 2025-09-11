"""Redis Stack 8 client with vector capabilities for EOL RAG Context.

This module provides a comprehensive Redis Stack vector storage client that supports
hierarchical document indexing, vector similarity search, and document management.
It integrates with Redis Search and RedisJSON modules to provide fast, scalable
vector operations with HNSW (Hierarchical Navigable Small World) indexing.

Key Features:
    - Hierarchical vector indexing (concepts → sections → chunks)
    - HNSW algorithm for efficient similarity search
    - Async/sync connection support with connection pooling
    - Multi-level document organization with parent-child relationships
    - Optimized vector storage and retrieval operations
    - Redis Search integration with filtering capabilities

Example:
    Basic usage:

    >>> from eol.rag_context.redis_client import RedisVectorStore
    >>> from eol.rag_context.config import RedisConfig, IndexConfig
    >>>
    >>> # Initialize store
    >>> redis_config = RedisConfig(host="localhost")
    >>> index_config = IndexConfig()
    >>> store = RedisVectorStore(redis_config, index_config)
    >>>
    >>> # Connect and create indexes
    >>> await store.connect_async()
    >>> store.create_hierarchical_indexes(embedding_dim=384)
    >>>
    >>> # Store document
    >>> doc = VectorDocument(
    ...     id="doc1",
    ...     content="Sample content",
    ...     embedding=np.random.rand(384),
    ...     hierarchy_level=3
    ... )
    >>> await store.store_document(doc)

"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from redis import Redis
    from redis.asyncio import Redis as AsyncRedis
    from redis.commands.search.field import NumericField, TagField, TextField, VectorField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
except ImportError as e:
    import sys

    error_msg = f"""
CRITICAL ERROR: Redis module not available!

The redis package with search support is required but not installed.
Error: {e}

To fix this, run:
    uv pip install redis[search]

Or if using pip:
    pip install redis[search]

Python executable: {sys.executable}
Python path: {sys.path}
"""
    print(error_msg, file=sys.stderr)
    sys.exit(1)

from .config import IndexConfig, RedisConfig

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Document with vector embedding for hierarchical storage.

    Represents a document or document fragment with its vector embedding and
    hierarchical metadata. Documents can be organized in a three-level hierarchy:
    concepts (level 1), sections (level 2), and chunks (level 3).

    Attributes:
        id: Unique identifier for the document.
        content: Text content of the document.
        embedding: Vector embedding as numpy array (typically float32).
        metadata: Additional metadata as key-value pairs.
        hierarchy_level: Level in hierarchy (1=concept, 2=section, 3=chunk).
        parent_id: Optional ID of parent document in hierarchy.
        children_ids: List of child document IDs.

    Example:
        Creating a chunk-level document:

        >>> import numpy as np
        >>> doc = VectorDocument(
        ...     id="chunk_001",
        ...     content="This is a text chunk.",
        ...     embedding=np.random.rand(384).astype(np.float32),
        ...     hierarchy_level=3,
        ...     parent_id="section_001",
        ...     metadata={"source": "doc.md", "position": 0}
        ... )
        >>> print(f"Document level: {doc.hierarchy_level}")
        Document level: 3

    """

    id: str
    content: str
    embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    hierarchy_level: int = 1  # 1=concept, 2=section, 3=chunk
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)


class RedisVectorStore:
    """Redis Stack vector store with hierarchical indexing and HNSW search.

    Provides comprehensive vector storage and retrieval using Redis Stack with
    vector search capabilities. Supports hierarchical document organization,
    efficient similarity search using HNSW algorithm, and both synchronous
    and asynchronous operations.

    The store organizes documents in a three-level hierarchy:
    1. Concepts: High-level topics and themes
    2. Sections: Mid-level document sections
    3. Chunks: Fine-grained text chunks

    Each level has optimized index settings for different query patterns:
    - Concepts: High precision for topic discovery
    - Sections: Balanced precision/recall for context retrieval
    - Chunks: High recall for detailed information

    Attributes:
        redis_config: Redis connection configuration.
        index_config: Vector index configuration parameters.
        redis: Synchronous Redis client instance.
        async_redis: Asynchronous Redis client instance.

    Example:
        Basic setup and usage:

        >>> from eol.rag_context.config import RedisConfig, IndexConfig
        >>>
        >>> # Configure connections
        >>> redis_config = RedisConfig(
        ...     host="localhost",
        ...     port=6379,
        ...     max_connections=50
        ... )
        >>> index_config = IndexConfig(
        ...     algorithm="HNSW",
        ...     distance_metric="COSINE",
        ...     m=16
        ... )
        >>>
        >>> # Initialize store
        >>> store = RedisVectorStore(redis_config, index_config)
        >>> await store.connect_async()
        >>>
        >>> # Create hierarchical indexes
        >>> store.create_hierarchical_indexes(embedding_dim=384)
        >>> print("Indexes created successfully")
        Indexes created successfully

    """

    def __init__(self, redis_config: RedisConfig, index_config: IndexConfig):
        """Initialize Redis vector store with configuration.

        Args:
            redis_config: Redis connection configuration including host, port,
                authentication, and connection pooling settings.
            index_config: Vector index configuration including algorithm settings,
                distance metrics, and hierarchical prefixes.

        """
        self.redis_config = redis_config
        self.index_config = index_config
        self.redis: Redis | None = None
        self.async_redis: AsyncRedis | None = None

    def connect(self) -> None:
        """Establish synchronous Redis connection with optimized settings.

        Creates a synchronous Redis connection using connection pooling for
        optimal performance. Configures platform-specific socket options and
        validates connectivity with a ping test.

        The connection uses the following optimizations:
        - Connection pooling for reduced overhead
        - TCP keepalive for connection health (non-macOS)
        - Binary mode for vector data preservation
        - Configurable authentication and database selection

        Raises:
            redis.ConnectionError: If unable to connect to Redis server.
            redis.AuthenticationError: If authentication fails.
            redis.RedisError: For other Redis-related connection issues.

        Example:
            >>> store = RedisVectorStore(redis_config, index_config)
            >>> store.connect()
            >>> print("Connected successfully")
            Connected successfully

        """
        # Build connection parameters for sync Redis client
        connection_kwargs = {
            "host": self.redis_config.host,
            "port": self.redis_config.port,
            "db": self.redis_config.db,
            "password": self.redis_config.password,
            "decode_responses": self.redis_config.decode_responses,
            "max_connections": self.redis_config.max_connections,
        }

        # Add platform-specific socket keepalive options
        if self.redis_config.socket_keepalive:
            connection_kwargs["socket_keepalive"] = True
            # Set platform-specific socket options
            import platform
            import socket

            if platform.system() == "Linux":
                # Use Linux-specific TCP keepalive options
                keepalive_options = {}
                if hasattr(socket, "TCP_KEEPIDLE"):
                    keepalive_options[socket.TCP_KEEPIDLE] = 1
                if hasattr(socket, "TCP_KEEPINTVL"):
                    keepalive_options[socket.TCP_KEEPINTVL] = 3
                if hasattr(socket, "TCP_KEEPCNT"):
                    keepalive_options[socket.TCP_KEEPCNT] = 5
                if keepalive_options:
                    connection_kwargs["socket_keepalive_options"] = keepalive_options
            # Skip socket options on macOS and other platforms to avoid issues

        self.redis = Redis(**connection_kwargs)

        # Validate connection with ping test
        try:
            self.redis.ping()
            logger.info(f"Connected to Redis at {self.redis_config.host}:{self.redis_config.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def connect_async(self) -> None:
        """Establish asynchronous Redis connection with connection pooling.

        Creates an asynchronous Redis connection optimized for high-throughput
        vector operations. Uses connection pooling and platform-specific
        optimizations for best performance in async environments.

        The async connection provides:
        - Non-blocking I/O for concurrent operations
        - Connection pooling for optimal resource usage
        - Automatic reconnection handling
        - Platform-specific socket optimizations (excluding macOS)

        Raises:
            redis.ConnectionError: If unable to establish async connection.
            redis.AuthenticationError: If Redis authentication fails.
            redis.RedisError: For other async Redis connection issues.

        Example:
            >>> store = RedisVectorStore(redis_config, index_config)
            >>> await store.connect_async()
            >>> print("Async connection established")
            Async connection established

        """
        # Build connection parameters for async Redis client
        async_connection_kwargs = {
            "host": self.redis_config.host,
            "port": self.redis_config.port,
            "db": self.redis_config.db,
            "password": self.redis_config.password,
            "decode_responses": self.redis_config.decode_responses,
            "max_connections": self.redis_config.max_connections,
        }

        # Configure socket keepalive for non-macOS platforms only
        if self.redis_config.socket_keepalive:
            async_connection_kwargs["socket_keepalive"] = True
            # Set platform-specific socket options
            import platform
            import socket

            if platform.system() == "Linux":
                # Use Linux-specific TCP keepalive options
                keepalive_options = {}
                if hasattr(socket, "TCP_KEEPIDLE"):
                    keepalive_options[socket.TCP_KEEPIDLE] = 1
                if hasattr(socket, "TCP_KEEPINTVL"):
                    keepalive_options[socket.TCP_KEEPINTVL] = 3
                if hasattr(socket, "TCP_KEEPCNT"):
                    keepalive_options[socket.TCP_KEEPCNT] = 5
                if keepalive_options:
                    async_connection_kwargs["socket_keepalive_options"] = keepalive_options
            # Skip socket options on macOS and other platforms to avoid issues

        self.async_redis = AsyncRedis(**async_connection_kwargs)

        # Validate async connection with ping test
        try:
            await self.async_redis.ping()
            logger.info(
                f"Async connected to Redis at {self.redis_config.host}:{self.redis_config.port}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def create_hierarchical_indexes(self, embedding_dim: int) -> None:
        """Create optimized vector indexes for each hierarchy level.

        Creates three specialized vector indexes using HNSW algorithm, each optimized
        for different search patterns and document types:

        1. Concept Index: High precision for topic discovery and concept mapping
        2. Section Index: Balanced precision/recall for contextual information
        3. Chunk Index: High recall using FLAT algorithm for detailed search

        Each index uses different HNSW parameters tuned for the expected
        query patterns and document sizes at that hierarchy level.

        Args:
            embedding_dim: Dimension of the vector embeddings (e.g., 384, 768, 1536).

        Raises:
            redis.ResponseError: If index creation fails due to Redis configuration.
            ValueError: If embedding dimension is invalid or unsupported.

        Example:
            >>> store = RedisVectorStore(redis_config, index_config)
            >>> store.connect()
            >>> store.create_hierarchical_indexes(embedding_dim=384)
            >>> print("All indexes created")
            All indexes created

        Note:
            - Concepts use HNSW with M=16, EF_CONSTRUCTION=200 for precision
            - Sections use HNSW with M=24, EF_CONSTRUCTION=300 for balance
            - Chunks use FLAT algorithm for exact nearest neighbor search

        """

        # Define schemas for each level
        schemas = {
            "concept": {
                "prefix": self.index_config.concept_prefix,
                "fields": [
                    TextField("content", weight=1.0),
                    VectorField(
                        "embedding",
                        self.index_config.algorithm,
                        {
                            "TYPE": "FLOAT32",
                            "DIM": embedding_dim,
                            "DISTANCE_METRIC": self.index_config.distance_metric,
                            "INITIAL_CAP": self.index_config.initial_cap,
                            "M": self.index_config.m,
                            "EF_CONSTRUCTION": self.index_config.ef_construction,
                        },
                    ),
                    TagField("children"),
                    NumericField("created_at"),
                    TextField("metadata"),
                ],
            },
            "section": {
                "prefix": self.index_config.section_prefix,
                "fields": [
                    TextField("content", weight=0.8),
                    VectorField(
                        "embedding",
                        self.index_config.algorithm,
                        {
                            "TYPE": "FLOAT32",
                            "DIM": embedding_dim,
                            "DISTANCE_METRIC": self.index_config.distance_metric,
                            "INITIAL_CAP": self.index_config.initial_cap * 10,
                            "M": self.index_config.m + 8,
                            "EF_CONSTRUCTION": self.index_config.ef_construction + 100,
                        },
                    ),
                    TagField("parent"),
                    TagField("children"),
                    NumericField("created_at"),
                    TextField("metadata"),
                ],
            },
            "chunk": {
                "prefix": self.index_config.chunk_prefix,
                "fields": [
                    TextField("content", weight=0.6),
                    VectorField(
                        "embedding",
                        "FLAT",  # Use FLAT for chunk level (exact search)
                        {
                            "TYPE": "FLOAT32",
                            "DIM": embedding_dim,
                            "DISTANCE_METRIC": self.index_config.distance_metric,
                            "INITIAL_CAP": min(self.index_config.initial_cap * 10, 100000),
                        },
                    ),
                    TagField("parent"),
                    NumericField("position"),
                    NumericField("created_at"),
                    TextField("metadata"),
                    TagField("doc_type"),  # markdown, code, pdf, etc.
                    TagField("language"),  # for code files
                ],
            },
        }

        # Create indexes
        for level_name, schema in schemas.items():
            index_name = f"{self.index_config.index_name}_{level_name}"

            try:
                # Check if index exists
                self.redis.ft(index_name).info()
                logger.info(f"Index {index_name} already exists")
            except Exception:
                # Create new index
                definition = IndexDefinition(prefix=[schema["prefix"]], index_type=IndexType.HASH)

                self.redis.ft(index_name).create_index(
                    fields=schema["fields"], definition=definition
                )
                logger.info(f"Created index {index_name}")

    async def store_document(self, doc: VectorDocument) -> None:
        """Store document with hierarchical metadata and vector embedding.

        Stores a document in Redis with its vector embedding and hierarchical
        metadata. The document is stored in the appropriate index based on its
        hierarchy level, with optimized field structures for each level.

        The storage format includes:
        - Vector embedding as binary float32 data
        - Full text content for retrieval
        - JSON-encoded metadata with timestamps
        - Parent-child relationship links
        - Level-specific fields (position, doc_type, language for chunks)

        Args:
            doc: VectorDocument containing content, embedding, and metadata.

        Raises:
            redis.ConnectionError: If not connected to Redis.
            redis.DataError: If document data is invalid or corrupted.
            ValueError: If hierarchy_level is not 1, 2, or 3.

        Example:
            Storing a chunk document:

            >>> import numpy as np
            >>> doc = VectorDocument(
            ...     id="chunk_123",
            ...     content="This is sample text content.",
            ...     embedding=np.random.rand(384).astype(np.float32),
            ...     hierarchy_level=3,
            ...     parent_id="section_45",
            ...     metadata={
            ...         "source": "readme.md",
            ...         "position": 2,
            ...         "doc_type": "markdown",
            ...         "language": "en"
            ...     }
            ... )
            >>> await store.store_document(doc)
            >>> print("Document stored successfully")
            Document stored successfully

        """
        if not self.async_redis:
            await self.connect_async()

        # Determine prefix based on hierarchy level
        prefix_map = {
            1: self.index_config.concept_prefix,
            2: self.index_config.section_prefix,
            3: self.index_config.chunk_prefix,
        }
        prefix = prefix_map.get(doc.hierarchy_level, self.index_config.chunk_prefix)

        # Prepare data for storage
        key = f"{prefix}{doc.id}"
        data = {
            "content": doc.content,
            "embedding": doc.embedding.astype(np.float32).tobytes(),
            "metadata": json.dumps(doc.metadata),
            "created_at": doc.metadata.get("created_at", 0),
        }

        # Add hierarchy-specific fields
        if doc.parent_id:
            data["parent"] = doc.parent_id
        if doc.children_ids:
            data["children"] = ",".join(doc.children_ids)
        if doc.hierarchy_level == 3:  # Chunk level
            data["position"] = doc.metadata.get("position", 0)
            data["doc_type"] = doc.metadata.get("doc_type", "unknown")
            data["language"] = doc.metadata.get("language", "unknown")

        # Store in Redis
        await self.async_redis.hset(key, mapping=data)
        logger.debug(f"Stored document {key}")

    async def vector_search(
        self,
        query_embedding: np.ndarray,
        hierarchy_level: int = 3,
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Perform vector similarity search at specified hierarchy level.

        Executes a vector similarity search using Redis Search with KNN queries.
        Searches within the specified hierarchy level using the appropriate index
        and returns ranked results by similarity score.

        The search uses cosine similarity by default and supports filtering by
        metadata fields such as source, document type, or parent relationships.

        Args:
            query_embedding: Query vector as numpy array (float32 recommended).
            hierarchy_level: Level to search (1=concept, 2=section, 3=chunk).
            k: Maximum number of results to return.
            filters: Optional filters as field-value pairs for TAG/NUMERIC fields.

        Returns:
            List of tuples containing (document_id, similarity_score, document_data).
            Scores are sorted in ascending order (lower = more similar for cosine).

        Raises:
            redis.ConnectionError: If not connected to Redis.
            redis.ResponseError: If search query is malformed.
            ValueError: If hierarchy_level is not 1, 2, or 3.

        Example:
            Basic similarity search:

            >>> import numpy as np
            >>> query_vec = np.random.rand(384).astype(np.float32)
            >>> results = await store.vector_search(
            ...     query_embedding=query_vec,
            ...     hierarchy_level=3,
            ...     k=5
            ... )
            >>> for doc_id, score, data in results:
            ...     print(f"{doc_id}: {score:.3f} - {data['content'][:50]}...")

            Search with filters:

            >>> results = await store.vector_search(
            ...     query_embedding=query_vec,
            ...     hierarchy_level=3,
            ...     k=10,
            ...     filters={"doc_type": "markdown", "language": "en"}
            ... )

        """
        if not self.async_redis:
            await self.connect_async()

        # Select appropriate index
        level_map = {1: "concept", 2: "section", 3: "chunk"}
        index_name = f"{self.index_config.index_name}_{level_map[hierarchy_level]}"

        # Build query
        query_vector = query_embedding.astype(np.float32).tobytes()

        # Base KNN query
        query_str = f"*=>[KNN {k} @embedding $vec AS score]"

        # Add filters if provided
        if filters:
            filter_clauses = []
            for field_name, value in filters.items():
                if isinstance(value, str):
                    # For TAG fields, escape special characters properly
                    # Redis TAGs need values wrapped in { } with no escaping inside
                    filter_clauses.append(f"@{field_name}:{{{value}}}")
                elif isinstance(value, int | float):
                    filter_clauses.append(f"@{field_name}:[{value} {value}]")

            if filter_clauses:
                query_str = f"({' '.join(filter_clauses)}) {query_str}"

        # Create query
        query = (
            Query(query_str)
            .return_fields("content", "metadata", "score", "parent", "children")
            .sort_by("score", asc=True)
            .dialect(2)
        )

        # Execute search
        results = await self.async_redis.ft(index_name).search(
            query, query_params={"vec": query_vector}
        )

        # Parse results
        output = []
        for doc in results.docs:
            doc_id = doc.id.split(":")[-1]
            score = float(doc.score) if hasattr(doc, "score") else 0.0

            data = {
                "content": doc.content if hasattr(doc, "content") else "",
                "metadata": (json.loads(doc.metadata) if hasattr(doc, "metadata") else {}),
                "parent": doc.parent if hasattr(doc, "parent") else None,
                "children": doc.children.split(",") if hasattr(doc, "children") else [],
            }

            output.append((doc_id, score, data))

        return output

    async def hierarchical_search(
        self,
        query_embedding: np.ndarray,
        max_chunks: int = 10,
        strategy: str = "adaptive",
    ) -> list[dict[str, Any]]:
        """Perform hierarchical search starting from concepts down to chunks.

        Executes a multi-level search strategy that begins with concept-level
        similarity search, then drills down through sections to find the most
        relevant chunks. This approach provides better context-aware results
        by considering document structure and relationships.

        The search process:
        1. Find top 3 relevant concepts using vector similarity
        2. Search for sections within those concepts
        3. Retrieve chunks from the most relevant sections
        4. Score results using weighted combination of level similarities

        Args:
            query_embedding: Query vector as numpy array (float32 recommended).
            max_chunks: Maximum number of chunk results to return.
            strategy: Search strategy - "adaptive" for balanced results,
                "detailed" for more chunk-level results.

        Returns:
            List of dictionaries containing:
            - id: Document/chunk identifier
            - score: Combined hierarchical similarity score
            - content: Text content of the result
            - metadata: Document metadata and hierarchy info
            - hierarchy: Dict with concept/section/chunk IDs

        Example:
            >>> import numpy as np
            >>> query_vec = np.random.rand(384).astype(np.float32)
            >>> results = await store.hierarchical_search(
            ...     query_embedding=query_vec,
            ...     max_chunks=8,
            ...     strategy="detailed"
            ... )
            >>> for result in results:
            ...     print(f"Score: {result['score']:.3f}")
            ...     print(f"Content: {result['content'][:100]}...")
            ...     print(f"Hierarchy: {result['hierarchy']}")
            ...     print()

        Note:
            This method provides fallback to direct chunk search if no
            concepts are found, ensuring robust retrieval in all scenarios.

        """
        results = []

        # Step 1: Find relevant concepts
        concepts = await self.vector_search(query_embedding, hierarchy_level=1, k=3)

        if not concepts:
            # Fallback to direct chunk search
            chunks = await self.vector_search(query_embedding, hierarchy_level=3, k=max_chunks)
            return [{"id": c[0], "score": c[1], **c[2]} for c in chunks]

        # Step 2: Find sections within concepts
        sections = []
        for concept_id, concept_score, _concept_data in concepts:
            # For now, search without parent filter and manually filter results
            # TODO: Fix Redis TAG field filtering with KNN queries
            concept_sections = await self.vector_search(
                query_embedding,
                hierarchy_level=2,
                k=20,  # Get more results to filter manually
            )

            for sec_id, sec_score, sec_data in concept_sections:
                sections.append(
                    {
                        "id": sec_id,
                        "score": sec_score * 0.8 + concept_score * 0.2,  # Weighted score
                        "data": sec_data,
                        "concept_id": concept_id,
                    }
                )

        # Sort sections by combined score
        sections.sort(key=lambda x: x["score"], reverse=True)
        sections = sections[:10]  # Keep top sections

        # Step 3: Get specific chunks if needed
        if strategy == "detailed" or len(sections) < 3:
            for section in sections[:5]:
                # For now, search without parent filter
                # TODO: Fix Redis TAG field filtering with KNN queries
                section_chunks = await self.vector_search(
                    query_embedding, hierarchy_level=3, k=10  # Get more results
                )

                for chunk_id, chunk_score, chunk_data in section_chunks:
                    results.append(
                        {
                            "id": chunk_id,
                            "score": chunk_score * 0.6 + section["score"] * 0.4,
                            "content": chunk_data["content"],
                            "metadata": chunk_data["metadata"],
                            "hierarchy": {
                                "concept": concept_id,
                                "section": section["id"],
                                "chunk": chunk_id,
                            },
                        }
                    )
        else:
            # Return section-level content
            results = sections

        # Sort by final score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_chunks]

    async def get_document_tree(self, doc_id: str) -> dict[str, Any]:
        """Get full document tree from any node in the hierarchy.

        Recursively retrieves the complete document hierarchy starting from
        the specified document ID. Traverses both parent and child relationships
        to build a comprehensive tree structure showing the document's context
        within the overall hierarchy.

        This method is useful for understanding document relationships,
        reconstructing original document structure, and providing contextual
        information for retrieved results.

        Args:
            doc_id: Unique identifier of the document to start traversal from.

        Returns:
            Dictionary containing:
            - id: The document identifier
            - content: Document text content
            - metadata: Document metadata as parsed JSON
            - parent: Parent document ID (if exists)
            - parent_data: Recursive parent tree structure
            - children: List of child document trees
            - error: Error message if document not found

        Example:
            >>> tree = await store.get_document_tree("section_123")
            >>> print(f"Document: {tree['id']}")
            >>> print(f"Content: {tree['content'][:50]}...")
            >>> if 'parent_data' in tree:
            ...     print(f"Parent: {tree['parent_data']['id']}")
            >>> print(f"Children: {len(tree.get('children', []))}")

        Note:
            This method searches across all hierarchy levels to locate the
            document, making it flexible but potentially slower for large trees.

        """
        if not self.async_redis:
            await self.connect_async()

        # Try each prefix to find the document
        for prefix in [
            self.index_config.concept_prefix,
            self.index_config.section_prefix,
            self.index_config.chunk_prefix,
        ]:
            key = f"{prefix}{doc_id}"
            data = await self.async_redis.hgetall(key)

            if data:
                result = {
                    "id": doc_id,
                    "content": data.get(b"content", b"").decode(),
                    "metadata": json.loads(data.get(b"metadata", b"{}").decode()),
                }

                # Add parent if exists
                if b"parent" in data:
                    result["parent"] = data[b"parent"].decode()
                    parent_tree = await self.get_document_tree(result["parent"])
                    result["parent_data"] = parent_tree

                # Add children if exist
                if b"children" in data:
                    children_ids = data[b"children"].decode().split(",")
                    result["children"] = []
                    for child_id in children_ids:
                        if child_id:
                            child_tree = await self.get_document_tree(child_id)
                            result["children"].append(child_tree)

                return result

        return {"id": doc_id, "error": "Not found"}

    async def close(self) -> None:
        """Close all Redis connections and clean up resources.

        Properly closes both synchronous and asynchronous Redis connections,
        ensuring connection pools are cleaned up and no resources are leaked.
        Should be called during application shutdown or when the store is
        no longer needed.

        Example:
            >>> store = RedisVectorStore(redis_config, index_config)
            >>> await store.connect_async()
            >>> # ... use store ...
            >>> await store.close()
            >>> print("Connections closed")
            Connections closed

        """
        if self.redis:
            self.redis.close()
        if self.async_redis:
            await self.async_redis.close()
