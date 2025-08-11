"""Semantic cache implementation targeting optimal 31% hit rate for RAG systems.

This module implements an intelligent semantic caching system designed to achieve
the research-backed optimal hit rate of 31% for LLM query responses. The cache
uses vector similarity matching to identify semantically similar queries and
reuse their cached responses, significantly reducing computation costs and latency.

Key Features:
    - Vector similarity-based cache matching using cosine distance
    - Adaptive threshold optimization targeting 31% hit rate
    - Redis-backed vector storage with HNSW indexing
    - Automatic cache size management with LRU-style eviction
    - Comprehensive statistics tracking and performance monitoring
    - TTL-based cache expiration for data freshness

The 31% target hit rate is based on research showing this provides optimal
balance between cache effectiveness and result freshness for semantic queries.

Example:
    Basic usage with semantic caching:
    
    >>> from eol.rag_context.semantic_cache import SemanticCache
    >>> from eol.rag_context.config import CacheConfig
    >>> 
    >>> # Configure cache with 31% target hit rate
    >>> config = CacheConfig(
    ...     enabled=True,
    ...     target_hit_rate=0.31,
    ...     similarity_threshold=0.95
    ... )
    >>> 
    >>> # Initialize cache
    >>> cache = SemanticCache(config, embedding_manager, redis_store)
    >>> await cache.initialize()
    >>> 
    >>> # Check for cached response
    >>> response = await cache.get("What is machine learning?")
    >>> if response is None:
    ...     # Generate new response
    ...     response = await generate_response(query)
    ...     await cache.set(query, response)
    >>> 
    >>> # Monitor performance
    >>> stats = cache.get_stats()
    >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
"""

import hashlib
import json
import time
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
import logging

from .config import CacheConfig
from .embeddings import EmbeddingManager
from .redis_client import RedisVectorStore

logger = logging.getLogger(__name__)


@dataclass
class CachedQuery:
    """Cached query entry with response, embedding, and usage metadata.
    
    Represents a single cached query-response pair with its vector embedding
    and tracking metadata. Used internally by the semantic cache to store
    and manage cached entries efficiently.
    
    Attributes:
        query: Original query string that was cached.
        response: Generated response text for the query.
        embedding: Vector embedding of the query for similarity matching.
        timestamp: Unix timestamp when the entry was created.
        hit_count: Number of times this cached entry has been retrieved.
        metadata: Additional metadata dictionary for extensibility.
        
    Example:
        Creating a cached query entry:
        
        >>> import numpy as np
        >>> import time
        >>> 
        >>> cached_entry = CachedQuery(
        ...     query="What is Python?",
        ...     response="Python is a programming language...",
        ...     embedding=np.random.rand(384).astype(np.float32),
        ...     timestamp=time.time(),
        ...     hit_count=0,
        ...     metadata={"source": "llm_response", "model": "claude-3"}
        ... )
        >>> print(f"Query: {cached_entry.query}")
        Query: What is Python?
    """

    query: str
    response: str
    embedding: np.ndarray
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticCache:
    """Intelligent semantic cache for LLM responses with adaptive optimization.
    
    Implements a sophisticated caching system that uses vector similarity to match
    semantically similar queries and reuse their cached responses. The cache is
    designed to achieve the research-backed optimal hit rate of 31% through
    adaptive threshold adjustment and intelligent cache management.
    
    Research shows that approximately 31% of queries in conversational AI systems
    can be effectively cached using semantic similarity, providing significant
    performance improvements while maintaining response quality and freshness.
    
    Key Components:
        - Vector-based similarity matching using embedding models
        - Adaptive threshold optimization for hit rate targeting
        - Redis-backed storage with HNSW vector indexing
        - LRU-style eviction with timestamp-based ordering
        - Comprehensive performance analytics and monitoring
    
    Attributes:
        config: Cache configuration including thresholds and limits.
        embeddings: Embedding manager for query vectorization.
        redis: Redis vector store for cache storage and search.
        stats: Dictionary tracking cache performance metrics.
        similarity_scores: List of recent similarity scores for analysis.
        adaptive_threshold: Current threshold adjusted for target hit rate.
        
    Example:
        Complete cache setup and usage:
        
        >>> from eol.rag_context.config import CacheConfig
        >>> 
        >>> # Configure for optimal performance
        >>> cache_config = CacheConfig(
        ...     enabled=True,
        ...     ttl_seconds=3600,  # 1 hour cache lifetime
        ...     similarity_threshold=0.95,
        ...     target_hit_rate=0.31,  # Research-backed optimum
        ...     adaptive_threshold=True,
        ...     max_cache_size=1000
        ... )
        >>> 
        >>> # Initialize cache system
        >>> cache = SemanticCache(cache_config, embedding_manager, redis_store)
        >>> await cache.initialize()
        >>> 
        >>> # Use in query processing
        >>> async def process_query(query: str) -> str:
        ...     # Check cache first
        ...     cached_response = await cache.get(query)
        ...     if cached_response:
        ...         return cached_response
        ...     
        ...     # Generate new response
        ...     response = await generate_llm_response(query)
        ...     
        ...     # Cache for future use
        ...     await cache.set(query, response)
        ...     return response
        >>> 
        >>> # Monitor performance
        >>> stats = cache.get_stats()
        >>> print(f"Cache hit rate: {stats['hit_rate']:.1%}")
        >>> print(f"Adaptive threshold: {stats['adaptive_threshold']:.3f}")
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        embedding_manager: EmbeddingManager,
        redis_store: RedisVectorStore,
    ):
        """Initialize semantic cache with configuration and dependencies.
        
        Args:
            cache_config: Configuration object containing cache settings including
                hit rate targets, similarity thresholds, and size limits.
            embedding_manager: Manager for generating and caching query embeddings.
            redis_store: Redis vector store for cache storage and similarity search.
        """
        self.config = cache_config
        self.embeddings = embedding_manager
        self.redis = redis_store

        # Cache performance statistics tracking
        self.stats = {
            "queries": 0,
            "hits": 0,
            "misses": 0,
            "hit_rate": 0.0,
            "avg_similarity": 0.0,
            "threshold_adjustments": 0,
        }

        # Adaptive threshold optimization
        self.similarity_scores: List[float] = []
        self.adaptive_threshold = cache_config.similarity_threshold

    async def initialize(self) -> None:
        """Initialize Redis vector index for semantic cache storage.
        
        Creates a dedicated Redis Search index optimized for semantic similarity
        queries. The index uses HNSW algorithm for efficient vector search and
        includes fields for query text, response, and metadata storage.
        
        Index Schema:
            - query (TextField): Original query string for exact matches
            - response (TextField): Cached response content
            - embedding (VectorField): Query vector for similarity search
            - timestamp (NumericField): Creation time for eviction policies
            - hit_count (NumericField): Usage tracking for popularity-based decisions
            - metadata (TextField): JSON-encoded additional metadata
        
        Raises:
            redis.ResponseError: If index creation fails due to Redis configuration.
            redis.ConnectionError: If unable to connect to Redis instance.
            
        Example:
            >>> cache = SemanticCache(config, embedding_manager, redis_store)
            >>> await cache.initialize()
            >>> print("Cache index ready for queries")
            Cache index ready for queries
        """
        try:
            # Create dedicated cache index
            await self.redis.async_redis.ft("cache_index").info()
            logger.info("Cache index already exists")
        except:
            # Create new cache index
            from redis.commands.search.field import VectorField, TextField, NumericField
            from redis.commands.search.index_definition import IndexDefinition, IndexType

            schema = [
                TextField("query"),
                TextField("response"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.embeddings.config.dimension,
                        "DISTANCE_METRIC": "COSINE",
                        "INITIAL_CAP": 1000,
                        "M": 16,
                        "EF_CONSTRUCTION": 200,
                    },
                ),
                NumericField("timestamp"),
                NumericField("hit_count"),
                TextField("metadata"),
            ]

            definition = IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH)

            await self.redis.async_redis.ft("cache_index").create_index(
                fields=schema, definition=definition
            )
            logger.info("Created cache index")

    async def get(self, query: str) -> Optional[str]:
        """Retrieve cached response for semantically similar query.
        
        Searches the cache for queries semantically similar to the input query
        using vector similarity matching. If a sufficiently similar cached query
        is found (above the similarity threshold), returns its cached response.
        
        The method implements adaptive threshold optimization, automatically
        adjusting the similarity threshold to maintain the target hit rate of 31%.
        All queries are tracked for performance analytics and optimization.
        
        Args:
            query: Input query string to search for in cache.
            
        Returns:
            Cached response string if similar query found above threshold,
            None if no suitable match exists or caching is disabled.
            
        Example:
            >>> # Check cache before expensive LLM call
            >>> response = await cache.get("What is machine learning?")
            >>> if response:
            ...     print("Cache hit!")
            ...     return response
            >>> else:
            ...     print("Cache miss, generating new response")
            ...     response = await generate_response(query)
            ...     await cache.set(query, response)
            
        Note:
            This method automatically updates cache statistics and hit counts
            for retrieved entries to support eviction and optimization algorithms.
        """
        self.stats["queries"] += 1

        if not self.config.enabled:
            self.stats["misses"] += 1
            return None

        # Get query embedding
        query_embedding = await self.embeddings.get_embedding(query)

        # Search for similar queries
        similar = await self._search_similar(query_embedding)

        if similar:
            best_match = similar[0]
            similarity = best_match[1]

            # Track similarity scores for adaptive threshold
            self.similarity_scores.append(similarity)

            # Check if similarity meets threshold
            threshold = (
                self.adaptive_threshold
                if self.config.adaptive_threshold
                else self.config.similarity_threshold
            )

            if similarity >= threshold:
                self.stats["hits"] += 1

                # Update hit count
                cache_key = f"cache:{best_match[0]}"
                self.redis.redis.hincrby(cache_key, "hit_count", 1)

                # Get response
                response = best_match[2]["response"]

                # Update statistics
                self._update_stats()

                logger.debug(f"Cache hit for query (similarity: {similarity:.3f})")
                return response

        self.stats["misses"] += 1
        self._update_stats()
        return None

    async def set(
        self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store query-response pair in cache with automatic size management.
        
        Caches a query-response pair with its vector embedding for future semantic
        matching. The method automatically manages cache size by evicting oldest
        entries when the cache reaches capacity limits.
        
        Each cached entry includes:
        - Original query text and response
        - Vector embedding of the query
        - Timestamp for age-based eviction
        - Hit count tracking for popularity metrics
        - Optional metadata for extensibility
        
        Args:
            query: Original query string to cache.
            response: Generated response to store.
            metadata: Optional additional metadata to store with the entry.
            
        Example:
            Caching an LLM response:
            
            >>> query = "Explain quantum computing"
            >>> response = "Quantum computing is a type of computation..."
            >>> metadata = {
            ...     "model": "claude-3-opus",
            ...     "tokens": 150,
            ...     "cost": 0.002
            ... }
            >>> await cache.set(query, response, metadata)
            >>> print("Response cached for future queries")
            
        Note:
            If caching is disabled in configuration, this method returns
            immediately without storing the entry.
        """
        if not self.config.enabled:
            return

        # Check cache size limit
        cache_size = await self._get_cache_size()
        if cache_size >= self.config.max_cache_size:
            await self._evict_oldest()

        # Get query embedding
        query_embedding = await self.embeddings.get_embedding(query)

        # Create cache entry
        cache_id = hashlib.md5(f"{query}:{time.time()}".encode()).hexdigest()
        cache_key = f"cache:{cache_id}"

        cache_data = {
            "query": query,
            "response": response,
            "embedding": query_embedding.astype(np.float32).tobytes(),
            "timestamp": time.time(),
            "hit_count": 0,
            "metadata": json.dumps(metadata or {}),
        }

        # Store in Redis
        self.redis.redis.hset(cache_key, mapping=cache_data)

        # Set TTL
        self.redis.redis.expire(cache_key, self.config.ttl_seconds)

        logger.debug(f"Cached response for query")

    async def _search_similar(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar cached queries."""
        from redis.commands.search.query import Query

        # Prepare query
        query_vector = query_embedding.astype(np.float32).tobytes()

        redis_query = (
            Query(f"*=>[KNN {k} @embedding $vec AS similarity]")
            .return_fields("query", "response", "similarity", "hit_count", "timestamp")
            .sort_by("similarity", asc=False)
            .dialect(2)
        )

        # Execute search
        try:
            results = self.redis.redis.ft("cache_index").search(
                redis_query, query_params={"vec": query_vector}
            )

            # Parse results
            output = []
            for doc in results.docs:
                doc_id = doc.id.split(":")[-1]
                similarity = 1.0 - float(doc.similarity) if hasattr(doc, "similarity") else 0.0

                data = {
                    "query": doc.query if hasattr(doc, "query") else "",
                    "response": doc.response if hasattr(doc, "response") else "",
                    "hit_count": int(doc.hit_count) if hasattr(doc, "hit_count") else 0,
                    "timestamp": float(doc.timestamp) if hasattr(doc, "timestamp") else 0,
                }

                output.append((doc_id, similarity, data))

            return output
        except Exception as e:
            logger.error(f"Error searching cache: {e}")
            return []

    async def _get_cache_size(self) -> int:
        """Get current cache size."""
        cursor = 0
        count = 0

        while True:
            cursor, keys = self.redis.redis.scan(cursor, match="cache:*", count=100)
            count += len(keys)

            if cursor == 0:
                break

        return count

    async def _evict_oldest(self) -> None:
        """Evict oldest cache entries to make room."""
        # Find oldest entries
        cursor = 0
        entries = []

        while True:
            cursor, keys = self.redis.redis.scan(cursor, match="cache:*", count=100)

            for key in keys:
                timestamp = self.redis.redis.hget(key, "timestamp")
                if timestamp:
                    entries.append((key, float(timestamp)))

            if cursor == 0:
                break

        # Sort by timestamp and evict oldest
        entries.sort(key=lambda x: x[1])

        # Evict 10% of cache
        evict_count = max(1, len(entries) // 10)
        for key, _ in entries[:evict_count]:
            self.redis.redis.delete(key)

        logger.debug(f"Evicted {evict_count} cache entries")

    def _update_stats(self) -> None:
        """Update cache statistics and adaptive threshold."""
        # Update hit rate
        if self.stats["queries"] > 0:
            self.stats["hit_rate"] = self.stats["hits"] / self.stats["queries"]

        # Update average similarity
        if self.similarity_scores:
            self.stats["avg_similarity"] = np.mean(self.similarity_scores[-100:])

        # Adaptive threshold adjustment
        if self.config.adaptive_threshold and len(self.similarity_scores) >= 100:
            current_hit_rate = self.stats["hit_rate"]
            target_hit_rate = self.config.target_hit_rate

            # Adjust threshold to reach target hit rate
            if abs(current_hit_rate - target_hit_rate) > 0.05:
                if current_hit_rate < target_hit_rate:
                    # Lower threshold to increase hits
                    self.adaptive_threshold *= 0.98
                else:
                    # Raise threshold to decrease hits
                    self.adaptive_threshold *= 1.02

                # Keep within bounds
                self.adaptive_threshold = max(0.85, min(0.99, self.adaptive_threshold))

                self.stats["threshold_adjustments"] += 1
                logger.info(
                    f"Adjusted cache threshold to {self.adaptive_threshold:.3f} (hit rate: {current_hit_rate:.3f})"
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.copy()
        stats["adaptive_threshold"] = self.adaptive_threshold
        stats["cache_enabled"] = self.config.enabled
        return stats

    async def clear(self) -> None:
        """Clear all cached entries."""
        cursor = 0
        deleted = 0

        while True:
            cursor, keys = self.redis.redis.scan(cursor, match="cache:*", count=100)

            if keys:
                self.redis.redis.delete(*keys)
                deleted += len(keys)

            if cursor == 0:
                break

        # Reset statistics
        self.stats = {
            "queries": 0,
            "hits": 0,
            "misses": 0,
            "hit_rate": 0.0,
            "avg_similarity": 0.0,
            "threshold_adjustments": 0,
        }
        self.similarity_scores = []
        self.adaptive_threshold = self.config.similarity_threshold

        logger.info(f"Cleared {deleted} cache entries")

    async def optimize(self) -> Dict[str, Any]:
        """
        Optimize cache for target hit rate.

        Returns optimization report.
        """
        report = {
            "current_hit_rate": self.stats["hit_rate"],
            "target_hit_rate": self.config.target_hit_rate,
            "current_threshold": self.adaptive_threshold,
            "recommendations": [],
        }

        # Analyze similarity distribution
        if len(self.similarity_scores) >= 100:
            percentiles = np.percentile(self.similarity_scores, [25, 50, 75, 90, 95])
            report["similarity_percentiles"] = {
                "25%": float(percentiles[0]),
                "50%": float(percentiles[1]),
                "75%": float(percentiles[2]),
                "90%": float(percentiles[3]),
                "95%": float(percentiles[4]),
            }

            # Recommend threshold based on target hit rate
            if self.config.target_hit_rate == 0.31:
                # 31% hit rate corresponds to ~69th percentile
                recommended_threshold = np.percentile(self.similarity_scores, 69)
            else:
                # General case
                target_percentile = (1 - self.config.target_hit_rate) * 100
                recommended_threshold = np.percentile(self.similarity_scores, target_percentile)

            report["recommended_threshold"] = float(recommended_threshold)

            if abs(recommended_threshold - self.adaptive_threshold) > 0.05:
                report["recommendations"].append(
                    f"Adjust threshold from {self.adaptive_threshold:.3f} to {recommended_threshold:.3f}"
                )

        # Check cache size
        cache_size = await self._get_cache_size()
        report["cache_size"] = cache_size

        if cache_size > self.config.max_cache_size * 0.9:
            report["recommendations"].append(
                f"Cache near capacity ({cache_size}/{self.config.max_cache_size}), consider increasing max_cache_size"
            )

        # Check TTL effectiveness
        if self.stats["queries"] > 1000:
            if self.stats["hit_rate"] < 0.1:
                report["recommendations"].append(
                    "Very low hit rate, consider increasing TTL or lowering similarity threshold"
                )
            elif self.stats["hit_rate"] > 0.5:
                report["recommendations"].append(
                    "Very high hit rate, could increase threshold for better precision"
                )

        return report
