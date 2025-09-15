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

    # Try to import legacy FT.SEARCH for backward compatibility
    # These are not actually used in Vector Sets implementation
    try:
        from redis.commands.search.field import (  # noqa: F401
            NumericField,
            TagField,
            TextField,
            VectorField,
        )
        from redis.commands.search.index_definition import IndexDefinition, IndexType  # noqa: F401
        from redis.commands.search.query import Query  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        # These imports are optional - Vector Sets don't need them
        pass
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
    """Redis 8.2+ vector store with hierarchical Vector Sets and SVS-VAMANA search.

    Provides comprehensive vector storage and retrieval using Redis 8.2's built-in
    Vector Sets with SVS-VAMANA algorithm. Supports hierarchical document organization,
    efficient similarity search with quantization support, and both synchronous
    and asynchronous operations.

    The store organizes documents in a three-level hierarchy using separate Vector Sets:
    1. Concepts: High-level topics and themes (concept_vectorset)
    2. Sections: Mid-level document sections (section_vectorset)
    3. Chunks: Fine-grained text chunks (chunk_vectorset)

    Each level uses optimized Vector Set parameters for different query patterns:
    - Concepts: High precision with higher EF values for topic discovery
    - Sections: Balanced precision/recall for context retrieval
    - Chunks: High recall with optimized quantization for detailed information

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
        """Prepare hierarchical Vector Sets for each hierarchy level.

        In Redis 8.2+, Vector Sets are created automatically on first VADD operation.
        This method prepares the Vector Set names and stores embedding dimensions
        for use during document storage.

        Three Vector Sets will be created automatically:
        1. Concept Vector Set: High precision for topic discovery and concept mapping
        2. Section Vector Set: Balanced precision/recall for contextual information
        3. Chunk Vector Set: High recall with optimized quantization for detailed search

        Args:
            embedding_dim: Dimension of the vector embeddings (e.g., 384, 768, 1536).

        Example:
            >>> store = RedisVectorStore(redis_config, index_config)
            >>> store.connect()
            >>> store.create_hierarchical_indexes(embedding_dim=384)
            >>> print("Vector Sets prepared")
            Vector Sets prepared

        Note:
            - Vector Sets use SVS-VAMANA algorithm automatically
            - Concepts use M=16, EF_CONSTRUCTION=200 for precision
            - Sections use M=24, EF_CONSTRUCTION=300 for balance
            - Chunks use default parameters with Q8 quantization

        """

        # Store embedding dimension for use in document storage
        self._embedding_dim = embedding_dim

        # Vector Set names for each hierarchy level
        vector_sets = {
            "concept": self.index_config.concept_vectorset,
            "section": self.index_config.section_vectorset,
            "chunk": self.index_config.chunk_vectorset,
        }

        logger.info(f"Vector Set names prepared for embedding dimension {embedding_dim}:")
        for level, vectorset_name in vector_sets.items():
            logger.info(f"  {level.capitalize()}: {vectorset_name}")

        logger.info("Vector Sets will be created automatically on first document storage")

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

        # Store document metadata in Redis Hash
        await self.async_redis.hset(key, mapping=data)

        # Add vector to appropriate Vector Set
        vectorset_map = {
            1: self.index_config.concept_vectorset,
            2: self.index_config.section_vectorset,
            3: self.index_config.chunk_vectorset,
        }
        vectorset_name = vectorset_map.get(doc.hierarchy_level, self.index_config.chunk_vectorset)

        # Ensure embedding is 1D array
        embedding_array = doc.embedding
        if embedding_array.ndim == 2:
            # Flatten 2D array to 1D (e.g., [1, 384] -> [384])
            embedding_array = embedding_array.flatten()

        # Convert embedding to float32 list for VADD command
        embedding_values = embedding_array.astype(np.float32).tolist()

        # Validate embedding values
        if not embedding_values or len(embedding_values) == 0:
            logger.error(f"Empty embedding for document {doc.id}")
            return

        # Check for invalid values (NaN, inf)
        if np.any(np.isnan(embedding_array)) or np.any(np.isinf(embedding_array)):
            logger.error(f"Invalid embedding values (NaN or inf) for document {doc.id}")
            return

        # Use VADD to add vector to Vector Set
        # Format: VADD key VALUES dim val1 val2 ... element [Q8|NOQUANT|BIN] [EF num] [M num]
        vadd_args = ["VADD", vectorset_name, "VALUES", str(len(embedding_values))]
        # Pass each float value as a separate argument
        for v in embedding_values:
            vadd_args.append(str(v))  # v is already a float from tolist()
        vadd_args.append(doc.id)

        # Add quantization parameter based on hierarchy level
        quantization = self.index_config.get_quantization_for_level(doc.hierarchy_level)
        if quantization == "Q8":
            vadd_args.append("Q8")
        elif quantization == "NOQUANT":
            vadd_args.append("NOQUANT")
        elif quantization == "BIN":
            vadd_args.append("BIN")
        else:
            vadd_args.append("Q8")  # Default fallback

        # Add level-specific parameters after quantization
        if doc.hierarchy_level == 1:  # Concept level - high precision
            vadd_args.extend(
                ["EF", str(self.index_config.ef_construction), "M", str(self.index_config.m)]
            )
        elif doc.hierarchy_level == 2:  # Section level - balanced
            vadd_args.extend(
                [
                    "EF",
                    str(self.index_config.ef_construction + 100),
                    "M",
                    str(self.index_config.m + 8),
                ]
            )
        # Chunk level uses default parameters

        # Execute VADD command
        try:
            await self.async_redis.execute_command(*vadd_args)
            logger.debug(f"Stored document {key} and added to Vector Set {vectorset_name}")
        except Exception as e:
            logger.error(f"VADD failed for doc {doc.id} in {vectorset_name}")
            logger.error(f"  Embedding shape: {embedding_array.shape}")
            logger.error(f"  List dimension: {len(embedding_values)}")
            logger.error(f"  Error: {e}")
            raise

    async def vector_search(
        self,
        query_embedding: np.ndarray,
        hierarchy_level: int = 3,
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Perform vector similarity search using Redis 8.2+ Vector Sets.

        Executes a vector similarity search using VSIM command against the appropriate
        Vector Set for the specified hierarchy level. Returns ranked results by
        similarity score using cosine distance.

        Args:
            query_embedding: Query vector as numpy array (float32 recommended).
            hierarchy_level: Level to search (1=concept, 2=section, 3=chunk).
            k: Maximum number of results to return.
            filters: Optional filters applied after vector search in application layer.

        Returns:
            List of tuples containing (document_id, similarity_score, document_data).
            Scores are cosine similarity scores (higher = more similar).

        Raises:
            redis.ConnectionError: If not connected to Redis.
            redis.ResponseError: If Vector Set doesn't exist or query is malformed.
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

        """
        if not self.async_redis:
            await self.connect_async()

        # Select appropriate Vector Set
        vectorset_map = {
            1: self.index_config.concept_vectorset,
            2: self.index_config.section_vectorset,
            3: self.index_config.chunk_vectorset,
        }
        vectorset_name = vectorset_map.get(hierarchy_level, self.index_config.chunk_vectorset)

        # Convert query embedding to list for VSIM command
        # Ensure embedding is 1D array
        if len(query_embedding.shape) > 1:
            query_values = query_embedding.flatten().astype(np.float32).tolist()
        else:
            query_values = query_embedding.astype(np.float32).tolist()

        # Build VSIM command with EF parameter based on hierarchy level
        # Format: VSIM key VALUES num val1 val2 ... [COUNT k] [WITHSCORES]
        vsim_args = ["VSIM", vectorset_name, "VALUES", str(len(query_values))]
        # Pass each float value as a separate argument
        for v in query_values:
            vsim_args.append(str(v))  # v is already a float from tolist()
        vsim_args.extend(["COUNT", str(k), "WITHSCORES"])

        # Add EF parameter for search quality
        if hierarchy_level == 1:  # Concept level - higher quality search
            vsim_args.extend(["EF", str(self.index_config.ef_runtime * 10)])
        elif hierarchy_level == 2:  # Section level - balanced search
            vsim_args.extend(["EF", str(self.index_config.ef_runtime * 5)])
        else:  # Chunk level - use default EF
            vsim_args.extend(["EF", str(self.index_config.ef_runtime)])

        try:
            # Execute VSIM command
            vsim_results = await self.async_redis.execute_command(*vsim_args)
        except Exception as e:
            if "VSET does not exist" in str(e):
                logger.warning(
                    f"Vector Set {vectorset_name} does not exist, returning empty results"
                )
                return []
            raise

        # Parse VSIM results (alternating element_id, score pairs)
        output = []
        if vsim_results:
            # Convert Redis bytes to strings/floats
            parsed_results = []
            for _i, item in enumerate(vsim_results):
                if isinstance(item, bytes):
                    parsed_results.append(item.decode())
                else:
                    parsed_results.append(item)

            # Process pairs of (element_id, score)
            for i in range(0, len(parsed_results), 2):
                if i + 1 < len(parsed_results):
                    element_id = parsed_results[i]
                    score = float(parsed_results[i + 1])

                    # Fetch document metadata from Redis hash
                    prefix_map = {
                        1: self.index_config.concept_prefix,
                        2: self.index_config.section_prefix,
                        3: self.index_config.chunk_prefix,
                    }
                    prefix = prefix_map.get(hierarchy_level, self.index_config.chunk_prefix)
                    doc_key = f"{prefix}{element_id}"

                    doc_data = await self.async_redis.hgetall(doc_key)
                    if doc_data:
                        # Convert bytes keys/values to strings (skip binary embedding data)
                        if isinstance(doc_data, dict) and doc_data:
                            data = {}
                            for k, v in doc_data.items():
                                key_str = k.decode() if isinstance(k, bytes) else k
                                # Skip decoding binary embedding data
                                if key_str == "embedding":
                                    continue  # Skip binary embedding data
                                try:
                                    val_str = v.decode() if isinstance(v, bytes) else v
                                    data[key_str] = val_str
                                except UnicodeDecodeError:
                                    # Skip binary data that can't be decoded
                                    continue
                        else:
                            data = {}

                        # Parse stored data
                        processed_data = {
                            "content": data.get("content", ""),
                            "metadata": json.loads(data.get("metadata", "{}")),
                            "parent": data.get("parent"),
                            "children": (
                                data.get("children", "").split(",") if data.get("children") else []
                            ),
                        }

                        # Apply filters if provided (application-level filtering)
                        if filters:
                            match = True
                            for field_name, value in filters.items():
                                if field_name in processed_data:
                                    if processed_data[field_name] != value:
                                        match = False
                                        break
                                elif field_name in processed_data["metadata"]:
                                    if processed_data["metadata"][field_name] != value:
                                        match = False
                                        break
                                else:
                                    match = False
                                    break
                            if not match:
                                continue

                        output.append((element_id, score, processed_data))

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
