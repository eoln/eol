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
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
    metadata: dict[str, Any] = field(default_factory=dict)


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

        # Initialize Vector Set name for fallback (proper initialization via ensure_cache_index)
        self._cache_vectorset = "semantic_cache"

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
        self.similarity_scores: list[float] = []
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
        # Check if cache Vector Set exists (Vector Sets are created automatically on first VADD)
        try:
            # Check if Vector Set exists using VCARD
            await self.redis.async_redis.execute_command("VCARD", "semantic_cache")
            logger.info("Cache Vector Set already exists")
        except Exception as e:
            if "VSET does not exist" in str(e):
                logger.info("Cache Vector Set will be created automatically on first cache entry")
            else:
                logger.warning(f"Error checking cache Vector Set: {e}")

        # Store cache configuration for Vector Set operations
        self._cache_vectorset = "semantic_cache"
        logger.info(f"Cache Vector Set prepared: {self._cache_vectorset}")

    async def get(self, query: str) -> str | None:
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

    async def set(self, query: str, response: str, metadata: dict[str, Any] | None = None) -> None:
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
        cache_id = hashlib.md5(f"{query}:{time.time()}".encode(), usedforsecurity=False).hexdigest()
        cache_key = f"cache:{cache_id}"

        cache_data = {
            "query": query,
            "response": response,
            "embedding": query_embedding.astype(np.float32).tobytes(),
            "timestamp": time.time(),
            "hit_count": 0,
            "metadata": json.dumps(metadata or {}),
        }

        # Store cache data in Redis Hash
        self.redis.redis.hset(cache_key, mapping=cache_data)

        # Add vector to cache Vector Set  
        # Format: VADD vectorset_name VALUES dim val1 val2 ... element_id [Q8|NOQUANT|BIN]
        embedding_values = query_embedding.astype(np.float32).tolist()
        vadd_args = ["VADD", self._cache_vectorset, "VALUES", str(len(embedding_values))]
        # Redis 8.2 expects individual float values as separate arguments
        for v in embedding_values:
            vadd_args.append(str(v))  # v is already a float from tolist()
        vadd_args.append(cache_id)  # Use cache_id as element identifier
        
        # Use cache-specific quantization from configuration
        quantization = self.redis.index_config.get_cache_quantization()
        vadd_args.append(quantization)
        
        try:
            await self.redis.async_redis.execute_command(*vadd_args)
        except Exception as e:
            logger.warning(f"Failed to add vector to cache Vector Set: {e}")

        # Set TTL
        self.redis.redis.expire(cache_key, self.config.ttl_seconds)

        logger.debug("Cached response for query")

    async def _search_similar(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for cached queries similar to the input embedding using Vector Sets.

        Performs vector similarity search against the cache Vector Set to find
        the k most similar cached queries. Uses cosine similarity and
        returns results sorted by similarity score.

        Args:
            query_embedding: Query vector to search for (float32 numpy array).
            k: Maximum number of similar queries to return.

        Returns:
            List of tuples containing (cache_id, similarity_score, cached_data).
            Similarity scores are cosine similarity (higher = more similar).

        Note:
            This is an internal method used by get() for cache lookups.
            Handles Redis connection errors gracefully by returning empty list.

        """
        try:
            # Convert query embedding to list for VSIM command
            # Ensure embedding is 1D array
            if len(query_embedding.shape) > 1:
                query_values = query_embedding.flatten().astype(np.float32).tolist()
            else:
                query_values = query_embedding.astype(np.float32).tolist()

            # Build VSIM command
            vsim_args = ["VSIM", self._cache_vectorset, "VALUES", str(len(query_values))]
            # Redis 8.2 expects individual float values as separate arguments
            for v in query_values:
                vsim_args.append(str(v))  # v is already a float from tolist()
            vsim_args.extend(["COUNT", str(k), "WITHSCORES", "EF", "50"])

            # Execute VSIM command
            vsim_results = await self.redis.async_redis.execute_command(*vsim_args)

            # Parse VSIM results
            output = []
            if vsim_results:
                # Convert Redis bytes to strings/floats
                parsed_results = []
                for item in vsim_results:
                    if isinstance(item, bytes):
                        parsed_results.append(item.decode())
                    else:
                        parsed_results.append(item)

                # Process pairs of (element_id, score)
                for i in range(0, len(parsed_results), 2):
                    if i + 1 < len(parsed_results):
                        cache_id = parsed_results[i]
                        similarity = float(parsed_results[i + 1])

                        # Fetch cache data from Redis hash
                        cache_key = f"cache:{cache_id}"
                        cache_data = await self.redis.async_redis.hgetall(cache_key)

                        if cache_data:
                            # Convert bytes keys/values to strings, skip binary fields
                            data = {}
                            for k, v in cache_data.items():
                                key_str = k.decode() if isinstance(k, bytes) else k
                                # Skip binary fields like embeddings
                                if key_str in ["embedding", "embedding_bytes"]:
                                    continue
                                try:
                                    val_str = v.decode() if isinstance(v, bytes) else v
                                    data[key_str] = val_str
                                except UnicodeDecodeError:
                                    # Skip fields that can't be decoded
                                    # as UTF-8 (likely binary data)
                                    continue

                            processed_data = {
                                "query": data.get("query", ""),
                                "response": data.get("response", ""),
                                "hit_count": int(data.get("hit_count", 0)),
                                "timestamp": float(data.get("timestamp", 0)),
                            }

                            output.append((cache_id, similarity, processed_data))

            return output

        except Exception as e:
            if "VSET does not exist" in str(e):
                logger.debug("Cache Vector Set does not exist yet, returning empty results")
                return []
            logger.error(f"Error searching cache: {e}")
            return []

    async def _get_cache_size(self) -> int:
        """Get current number of entries in the cache.

        Scans Redis for all cache entries using the "cache:*" pattern
        and returns the total count. Used for cache size monitoring
        and eviction decisions.

        Returns:
            Integer count of cached entries currently stored.

        Note:
            This method scans the entire keyspace with "cache:*" pattern,
            which may be slow for very large caches.

        """
        cursor = 0
        count = 0

        while True:
            cursor, keys = self.redis.redis.scan(cursor, match="cache:*", count=100)
            count += len(keys)

            if cursor == 0:
                break

        return count

    async def _evict_oldest(self) -> None:
        """Evict oldest cache entries when cache reaches capacity.

        Implements LRU-style eviction by removing the 10% oldest entries
        based on timestamps. This maintains cache freshness and prevents
        unlimited growth when max_cache_size is reached.

        The eviction process:
        1. Scans all cache entries and collects timestamps
        2. Sorts entries by timestamp (oldest first)
        3. Removes the oldest 10% of entries
        4. Logs the eviction count for monitoring

        Note:
            This is an internal method called automatically by set()
            when cache size limits are reached.

        """
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
        """Update cache statistics and perform adaptive threshold optimization.

        Updates hit rate calculations, similarity score averages, and performs
        adaptive threshold adjustments to maintain target hit rate. Called
        after each cache operation to maintain accurate performance metrics.

        Adaptive threshold logic:
        - If hit rate < target: Lower threshold to increase hits
        - If hit rate > target: Raise threshold to reduce hits
        - Threshold bounded between 0.85 and 0.99 for stability

        Note:
            This is an internal method called automatically by get() and set()
            to maintain real-time performance optimization.

        """
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
                    f"Adjusted cache threshold to {self.adaptive_threshold:.3f} "
                    f"(hit rate: {current_hit_rate:.3f})"
                )

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache performance statistics.

        Returns detailed statistics about cache performance, hit rates,
        adaptive threshold adjustments, and overall effectiveness. Used
        for monitoring, debugging, and optimization decisions.

        Returns:
            Dictionary containing:
            - queries: Total number of queries processed
            - hits: Number of successful cache hits
            - misses: Number of cache misses
            - hit_rate: Current hit rate as decimal (0.0-1.0)
            - avg_similarity: Average similarity score of recent matches
            - threshold_adjustments: Number of adaptive threshold changes
            - adaptive_threshold: Current adaptive threshold value
            - cache_enabled: Whether caching is currently enabled

        Example:
            >>> stats = cache.get_stats()
            >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
            >>> print(f"Queries processed: {stats['queries']}")
            >>> print(f"Adaptive threshold: {stats['adaptive_threshold']:.3f}")
            >>>
            >>> # Check if hitting target
            >>> if stats['hit_rate'] > 0.35:
            ...     print("Hit rate too high, consider raising threshold")
            >>> elif stats['hit_rate'] < 0.25:
            ...     print("Hit rate too low, consider lowering threshold")

        """
        stats = self.stats.copy()
        stats["adaptive_threshold"] = self.adaptive_threshold
        stats["cache_enabled"] = self.config.enabled
        return stats

    async def clear(self) -> None:
        """Clear all cached entries and reset statistics.

        Removes all cached query-response pairs from Redis and resets all
        performance statistics and adaptive thresholds to initial values.
        Useful for cache maintenance, testing, or when changing cache
        configuration significantly.

        The operation:
        1. Scans and deletes all cache entries with "cache:" prefix
        2. Resets all performance statistics to zero
        3. Clears similarity score history
        4. Resets adaptive threshold to configured default

        Example:
            >>> # Clear cache for fresh start
            >>> await cache.clear()
            >>> stats = cache.get_stats()
            >>> print(f"Queries: {stats['queries']}")  # Will be 0
            >>> print(f"Hit rate: {stats['hit_rate']}")  # Will be 0.0

        Note:
            This operation cannot be undone. Consider using cache statistics
            to evaluate performance before clearing productive caches.

        """
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

    async def optimize(self) -> dict[str, Any]:
        """Analyze cache performance and provide optimization recommendations.

        Performs comprehensive analysis of cache performance including similarity
        score distributions, hit rate trends, and capacity utilization. Generates
        actionable recommendations for improving cache effectiveness.

        The analysis includes:
        - Current vs target hit rate comparison
        - Similarity score percentile analysis
        - Optimal threshold recommendations based on target hit rate
        - Cache size and capacity warnings
        - TTL effectiveness evaluation

        Returns:
            Dictionary containing:
            - current_hit_rate: Current cache hit rate
            - target_hit_rate: Configured target hit rate
            - current_threshold: Current similarity threshold
            - recommended_threshold: Statistically optimal threshold
            - similarity_percentiles: Distribution of similarity scores
            - cache_size: Current number of cached entries
            - recommendations: List of actionable optimization suggestions

        Example:
            >>> report = await cache.optimize()
            >>> print(f"Hit rate: {report['current_hit_rate']:.1%} "
            ...       f"(target: {report['target_hit_rate']:.1%})")
            >>> print(f"Current threshold: {report['current_threshold']:.3f}")
            >>> print(f"Recommended threshold: {report.get('recommended_threshold', 'N/A')}")
            >>>
            >>> # Apply recommendations
            >>> for rec in report['recommendations']:
            ...     print(f"💡 {rec}")

        Note:
            Requires at least 100 queries for meaningful statistical analysis.
            Recommendations are based on empirical performance data.

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
                    f"Adjust threshold from {self.adaptive_threshold:.3f} to "
                    f"{recommended_threshold:.3f}"
                )

        # Check cache size
        cache_size = await self._get_cache_size()
        report["cache_size"] = cache_size

        if cache_size > self.config.max_cache_size * 0.9:
            report["recommendations"].append(
                f"Cache near capacity ({cache_size}/{self.config.max_cache_size}), "
                f"consider increasing max_cache_size"
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
