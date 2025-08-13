# Semantic Caching Patterns

## Cache Key Generation

### Semantic Similarity Approach
```python
async def get_cache_key(query: str, threshold: float = 0.85) -> Optional[str]:
    """Find semantically similar cached query"""
    query_embedding = await generate_embedding(query)
    
    # Search for similar queries in cache
    similar = await cache_store.search_similar(
        query_embedding, 
        threshold=threshold
    )
    
    if similar:
        return similar[0].cache_key
    return None
```

## Cache Storage Strategy
```python
class CacheEntry:
    query: str
    embedding: List[float]
    response: str
    timestamp: datetime
    hit_count: int
    ttl: int  # seconds
```

## TTL Management
- Short TTL (5 min): Rapidly changing data
- Medium TTL (1 hour): Standard queries
- Long TTL (24 hours): Reference data
- Infinite TTL: Static documentation

## Cache Hit Rate Optimization
Target: >31% hit rate

### Strategies:
1. **Query Normalization**
   - Remove stop words
   - Standardize formatting
   - Expand contractions

2. **Similarity Threshold Tuning**
   - Start with 0.85 similarity
   - Adjust based on hit/miss patterns
   - Monitor false positive rate

3. **Popular Query Preloading**
   - Identify common patterns
   - Pre-cache frequent queries
   - Warm cache on startup

## Performance Metrics
```python
# Track cache performance
metrics = {
    "total_queries": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "average_latency": 0,
    "hit_rate": 0.0
}
```