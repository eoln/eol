"""
Semantic cache implementation targeting 31% hit rate.
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
    """Cached query with response and metadata."""
    query: str
    response: str
    embedding: np.ndarray
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticCache:
    """
    Semantic cache for LLM responses targeting 31% hit rate.
    
    Based on research showing 31% of queries can be effectively cached
    with semantic similarity matching.
    """
    
    def __init__(
        self,
        cache_config: CacheConfig,
        embedding_manager: EmbeddingManager,
        redis_store: RedisVectorStore
    ):
        self.config = cache_config
        self.embeddings = embedding_manager
        self.redis = redis_store
        
        # Cache statistics
        self.stats = {
            "queries": 0,
            "hits": 0,
            "misses": 0,
            "hit_rate": 0.0,
            "avg_similarity": 0.0,
            "threshold_adjustments": 0
        }
        
        # Adaptive threshold tracking
        self.similarity_scores: List[float] = []
        self.adaptive_threshold = cache_config.similarity_threshold
    
    async def initialize(self) -> None:
        """Initialize cache index in Redis."""
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
                    }
                ),
                NumericField("timestamp"),
                NumericField("hit_count"),
                TextField("metadata"),
            ]
            
            definition = IndexDefinition(
                prefix=["cache:"],
                index_type=IndexType.HASH
            )
            
            await self.redis.async_redis.ft("cache_index").create_index(
                fields=schema,
                definition=definition
            )
            logger.info("Created cache index")
    
    async def get(self, query: str) -> Optional[str]:
        """
        Retrieve cached response for semantically similar query.
        
        Returns None if no sufficiently similar cached query found.
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
            threshold = self.adaptive_threshold if self.config.adaptive_threshold else self.config.similarity_threshold
            
            if similarity >= threshold:
                self.stats["hits"] += 1
                
                # Update hit count
                cache_key = f"cache:{best_match[0]}"
                await self.redis.redis.hincrby(cache_key, "hit_count", 1)
                
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
        self,
        query: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store query-response pair in cache."""
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
        await self.redis.redis.hset(cache_key, mapping=cache_data)
        
        # Set TTL
        await self.redis.redis.expire(cache_key, self.config.ttl_seconds)
        
        logger.debug(f"Cached response for query")
    
    async def _search_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 5
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
            results = await self.redis.redis.ft("cache_index").search(
                redis_query,
                query_params={"vec": query_vector}
            )
            
            # Parse results
            output = []
            for doc in results.docs:
                doc_id = doc.id.split(":")[-1]
                similarity = 1.0 - float(doc.similarity) if hasattr(doc, 'similarity') else 0.0
                
                data = {
                    "query": doc.query if hasattr(doc, 'query') else "",
                    "response": doc.response if hasattr(doc, 'response') else "",
                    "hit_count": int(doc.hit_count) if hasattr(doc, 'hit_count') else 0,
                    "timestamp": float(doc.timestamp) if hasattr(doc, 'timestamp') else 0,
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
            cursor, keys = await self.redis.redis.scan(
                cursor,
                match="cache:*",
                count=100
            )
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
            cursor, keys = await self.redis.redis.scan(
                cursor,
                match="cache:*",
                count=100
            )
            
            for key in keys:
                timestamp = await self.redis.redis.hget(key, "timestamp")
                if timestamp:
                    entries.append((key, float(timestamp)))
            
            if cursor == 0:
                break
        
        # Sort by timestamp and evict oldest
        entries.sort(key=lambda x: x[1])
        
        # Evict 10% of cache
        evict_count = max(1, len(entries) // 10)
        for key, _ in entries[:evict_count]:
            await self.redis.redis.delete(key)
        
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
                logger.info(f"Adjusted cache threshold to {self.adaptive_threshold:.3f} (hit rate: {current_hit_rate:.3f})")
    
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
            cursor, keys = await self.redis.redis.scan(
                cursor,
                match="cache:*",
                count=100
            )
            
            if keys:
                await self.redis.redis.delete(*keys)
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
            "threshold_adjustments": 0
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
            "recommendations": []
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