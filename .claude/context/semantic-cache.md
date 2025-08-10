# Semantic Caching Patterns for LLM Applications

## Overview
Semantic caching reduces LLM API calls by storing and retrieving similar queries based on semantic similarity rather than exact matches.

## Key Statistics
- **31% of queries** to LLMs can be served from cache
- **Response time**: Milliseconds (cache) vs seconds (LLM)
- **Cost reduction**: Up to 30% fewer API calls
- **Similarity threshold**: Typically 97% for high-confidence matches

## Core Implementation

### Basic Semantic Cache
```python
from redisvl import EmbeddingsCache
import hashlib
import time

class SemanticCache:
    def __init__(self, redis_client, similarity_threshold=0.97):
        self.redis = redis_client
        self.threshold = similarity_threshold
        self.embedder = EmbeddingModel()
        
    async def get_or_generate(self, query, generator):
        # Generate query embedding
        query_embedding = await self.embedder.encode(query)
        
        # Search for similar cached queries
        cached = await self.search_similar(query_embedding)
        
        if cached and cached.similarity >= self.threshold:
            # Cache hit - return stored response
            self.update_cache_stats(hit=True)
            return cached.response
        
        # Cache miss - generate new response
        response = await generator(query)
        
        # Store in cache
        await self.store(query, query_embedding, response)
        self.update_cache_stats(hit=False)
        
        return response
    
    async def search_similar(self, embedding):
        results = self.redis.ft().search(
            Query("*=>[KNN 1 @embedding $vec AS similarity]")
            .add_filter(f"@similarity >= {self.threshold}")
            .dialect(2),
            query_params={"vec": embedding.tobytes()}
        )
        
        if results.docs:
            return CachedResult(
                response=results.docs[0].response,
                similarity=float(results.docs[0].similarity)
            )
        return None
```

## Advanced Caching Strategies

### 1. Tiered Caching
```python
class TieredSemanticCache:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.tiers = [
            {"threshold": 0.99, "ttl": 86400},    # Exact match - 24h
            {"threshold": 0.97, "ttl": 3600},     # High similarity - 1h
            {"threshold": 0.95, "ttl": 900}       # Good similarity - 15m
        ]
    
    async def get_cached(self, query):
        embedding = await self.embed(query)
        
        for tier in self.tiers:
            result = await self.search_at_threshold(
                embedding, 
                tier["threshold"]
            )
            if result:
                # Validate TTL
                if self.is_valid(result, tier["ttl"]):
                    return result
                else:
                    # Expired - remove from cache
                    await self.invalidate(result.key)
        
        return None
```

### 2. Context-Aware Caching
```python
class ContextAwareCache:
    def __init__(self, redis_client):
        self.redis = redis_client
        
    async def cache_with_context(self, query, context, response):
        # Create composite key with context
        cache_key = self.generate_key(query, context)
        
        # Store with context metadata
        await self.redis.hset(cache_key, {
            "query": query,
            "context": json.dumps(context),
            "query_embedding": self.embed(query).tobytes(),
            "context_embedding": self.embed(str(context)).tobytes(),
            "response": response,
            "timestamp": time.time()
        })
    
    async def retrieve_with_context(self, query, context):
        # Search considering both query and context similarity
        query_emb = self.embed(query)
        context_emb = self.embed(str(context))
        
        # Weighted similarity search
        results = await self.weighted_search(
            query_emb, 
            context_emb,
            query_weight=0.7,
            context_weight=0.3
        )
        
        return results
```

### 3. Adaptive Caching
```python
class AdaptiveCache:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.hit_rate_target = 0.3  # 30% target hit rate
        
    async def adjust_threshold(self):
        """Dynamically adjust similarity threshold based on hit rate"""
        stats = await self.get_cache_stats()
        current_hit_rate = stats["hits"] / stats["total"]
        
        if current_hit_rate < self.hit_rate_target:
            # Lower threshold to increase hits
            self.threshold = max(0.90, self.threshold - 0.01)
        elif current_hit_rate > self.hit_rate_target + 0.1:
            # Raise threshold to improve quality
            self.threshold = min(0.99, self.threshold + 0.01)
        
        await self.redis.set("cache:threshold", self.threshold)
```

## Caching Patterns

### 1. FAQ/Documentation Cache
```python
class FAQCache:
    """Optimized for stable, high-frequency queries"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.persistent_mode = True
        
    async def preload_faqs(self, faq_data):
        """Preload common questions and answers"""
        for item in faq_data:
            embedding = self.embed(item["question"])
            
            await self.redis.hset(f"faq:{item['id']}", {
                "question": item["question"],
                "answer": item["answer"],
                "embedding": embedding.tobytes(),
                "confidence": 1.0,  # High confidence for curated content
                "persistent": True   # Never expire
            })
    
    async def query_faq(self, question):
        embedding = self.embed(question)
        
        # Search FAQs first with lower threshold
        result = await self.search_persistent(
            embedding, 
            threshold=0.95  # Lower threshold for FAQs
        )
        
        if result:
            return result.answer
        
        # Fallback to LLM
        return None
```

### 2. Session-Based Cache
```python
class SessionCache:
    """Cache per user session with conversation context"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.session_ttl = 1800  # 30 minutes
        
    async def cache_in_session(self, session_id, query, response):
        session_key = f"session:{session_id}"
        
        # Add to session history
        await self.redis.rpush(
            f"{session_key}:history",
            json.dumps({
                "query": query,
                "response": response,
                "timestamp": time.time()
            })
        )
        
        # Store with session-specific embedding
        embedding = await self.embed_with_history(
            query, 
            session_id
        )
        
        await self.redis.hset(f"{session_key}:cache", {
            hash(query): {
                "embedding": embedding.tobytes(),
                "response": response
            }
        })
        
        # Reset TTL
        await self.redis.expire(session_key, self.session_ttl)
```

### 3. Domain-Specific Cache
```python
class DomainCache:
    """Separate caches for different domains/topics"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.domains = {}
        
    async def register_domain(self, domain_name, config):
        """Register domain with specific settings"""
        self.domains[domain_name] = {
            "threshold": config.get("threshold", 0.97),
            "ttl": config.get("ttl", 3600),
            "embedder": config.get("embedder", "default"),
            "index": f"idx:cache:{domain_name}"
        }
        
        # Create domain-specific index
        await self.create_domain_index(domain_name)
    
    async def cache_by_domain(self, query, domain):
        if domain not in self.domains:
            raise ValueError(f"Unknown domain: {domain}")
        
        config = self.domains[domain]
        
        # Use domain-specific embedder
        embedding = await self.get_embedder(
            config["embedder"]
        ).encode(query)
        
        # Search in domain-specific index
        return await self.search_domain_cache(
            embedding,
            domain,
            config["threshold"]
        )
```

## Redis Implementation Details

### Index Creation
```python
async def create_cache_index(redis_client):
    """Create optimized index for semantic cache"""
    
    schema = {
        "index": {
            "name": "semantic_cache",
            "prefix": "cache:",
            "storage_type": "hash"
        },
        "fields": [
            {
                "name": "query",
                "type": "text",
                "weight": 1.0
            },
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "dims": 1536,
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                    "initial_cap": 10000,
                    "m": 16,
                    "ef_construction": 200
                }
            },
            {
                "name": "response",
                "type": "text",
                "no_index": True  # Don't index response content
            },
            {
                "name": "timestamp",
                "type": "numeric",
                "sortable": True
            },
            {
                "name": "hit_count",
                "type": "numeric",
                "sortable": True
            }
        ]
    }
    
    index = SearchIndex.from_dict(schema)
    await index.create(overwrite=False)
```

### Cache Invalidation
```python
class CacheInvalidator:
    def __init__(self, redis_client):
        self.redis = redis_client
        
    async def invalidate_by_age(self, max_age_seconds):
        """Remove entries older than max_age"""
        cutoff = time.time() - max_age_seconds
        
        await self.redis.ft().search(
            Query("@timestamp:[-inf {}]".format(cutoff))
            .return_fields("__key")
        ).delete()
    
    async def invalidate_by_pattern(self, pattern):
        """Invalidate entries matching pattern"""
        keys = await self.redis.keys(f"cache:{pattern}")
        if keys:
            await self.redis.delete(*keys)
    
    async def invalidate_low_confidence(self, min_confidence=0.95):
        """Remove low-confidence cached responses"""
        await self.redis.ft().search(
            Query(f"@confidence:[0 {min_confidence}]")
            .return_fields("__key")
        ).delete()
```

## Performance Optimization

### Batch Processing
```python
async def batch_cache_lookup(queries, redis_client):
    """Efficient batch cache lookups"""
    
    # Generate embeddings in batch
    embeddings = await batch_embed(queries)
    
    # Parallel cache lookups
    tasks = [
        search_cache(emb, redis_client) 
        for emb in embeddings
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Separate hits and misses
    hits = [(q, r) for q, r in zip(queries, results) if r]
    misses = [q for q, r in zip(queries, results) if not r]
    
    return hits, misses
```

### Memory Management
```python
class CacheMemoryManager:
    def __init__(self, redis_client, max_memory_mb=1000):
        self.redis = redis_client
        self.max_memory = max_memory_mb * 1024 * 1024
        
    async def evict_lru(self):
        """Evict least recently used entries"""
        memory_usage = await self.redis.memory_usage("cache:*")
        
        if memory_usage > self.max_memory:
            # Get LRU entries
            entries = await self.redis.ft().search(
                Query("*")
                .sort_by("timestamp", asc=True)
                .paging(0, 100)
            )
            
            # Delete oldest entries
            for entry in entries.docs:
                await self.redis.delete(entry.id)
                if await self.check_memory() < self.max_memory:
                    break
```

## Monitoring and Analytics

```python
class CacheAnalytics:
    def __init__(self, redis_client):
        self.redis = redis_client
        
    async def track_metrics(self):
        """Track cache performance metrics"""
        
        return {
            "hit_rate": await self.calculate_hit_rate(),
            "avg_similarity": await self.average_similarity(),
            "cache_size": await self.redis.dbsize(),
            "memory_usage": await self.redis.memory_usage("cache:*"),
            "response_time": {
                "cache": await self.avg_cache_response_time(),
                "llm": await self.avg_llm_response_time()
            },
            "cost_savings": await self.calculate_cost_savings()
        }
    
    async def calculate_cost_savings(self):
        """Estimate cost savings from caching"""
        stats = await self.get_stats()
        
        # Assuming $0.002 per 1K tokens
        avg_tokens_per_request = 500
        cost_per_request = (avg_tokens_per_request / 1000) * 0.002
        
        saved_requests = stats["cache_hits"]
        savings = saved_requests * cost_per_request
        
        return {
            "saved_requests": saved_requests,
            "estimated_savings_usd": savings
        }
```

## EOL Framework Integration

### Configuration
```yaml
# cache-config.eol
name: semantic-cache
phase: implementation

cache:
  type: semantic
  backend: redis
  settings:
    similarity_threshold: 0.97
    ttl: 3600
    max_memory_mb: 1000
    
  strategies:
    - tiered_caching
    - context_aware
    - domain_specific
    
  monitoring:
    enabled: true
    metrics_interval: 60
```

### Usage Example
```python
# EOL semantic cache integration
from eol.cache import SemanticCache

cache = SemanticCache(
    redis_url="redis://localhost:6379",
    threshold=0.97
)

async def handle_query(query):
    return await cache.get_or_generate(
        query,
        generator=lambda q: llm.generate(q)
    )
```

## Best Practices

1. **Set appropriate similarity thresholds**:
   - 0.99+ for exact matches
   - 0.97 for high-confidence semantic matches
   - 0.95 for exploratory/FAQ matching

2. **Implement TTL strategies**:
   - Permanent for curated content
   - 24h for general queries
   - 30m for session-based caching

3. **Monitor and adjust**:
   - Track hit rates
   - Adjust thresholds dynamically
   - Clean up stale entries

4. **Use domain-specific caches**:
   - Separate technical from general queries
   - Different thresholds per domain
   - Specialized embedders when needed