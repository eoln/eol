# quality-performance-check - RAG Performance Benchmarks

Specialized performance validation for RAG operations and Redis vector database performance.

## Purpose
Validate and benchmark RAG-specific performance metrics against targets.

## Usage
```bash
/quality:performance-check [--component=all|indexing|search|cache]
```

## Performance Targets

| Operation | Target | Critical |
|-----------|--------|----------|
| Document Indexing | >10 docs/sec | Yes |
| Vector Search | <100ms P95 | Yes |
| Cache Hit Rate | >31% | Yes |
| Embedding Generation | <50ms/chunk | No |
| Context Window Usage | <80% | No |
| Redis Memory | <2GB | No |

## Benchmark Components

### 1. Document Indexing Performance
```python
async def benchmark_indexing():
    """Test document indexing throughput"""
    docs = generate_test_documents(100)
    
    start = time.time()
    await indexer.index_documents(docs)
    duration = time.time() - start
    
    rate = len(docs) / duration
    print(f"Indexing rate: {rate:.1f} docs/sec")
    assert rate > 10, "Below target: 10 docs/sec"
```

### 2. Vector Search Latency
```python
async def benchmark_search():
    """Test vector search latency"""
    queries = generate_test_queries(100)
    latencies = []
    
    for query in queries:
        start = time.time()
        await redis_store.search_similar(query, k=5)
        latencies.append((time.time() - start) * 1000)
    
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    print(f"Search latency - P50: {p50:.1f}ms, P95: {p95:.1f}ms, P99: {p99:.1f}ms")
    assert p95 < 100, "P95 latency exceeds 100ms"
```

### 3. Semantic Cache Performance
```python
def benchmark_cache():
    """Test cache hit rate and performance"""
    # Warm up cache
    for query in training_queries:
        await cache.get_or_compute(query, compute_fn)
    
    # Test hit rate
    hits = 0
    for query in test_queries:
        if await cache.get(query):
            hits += 1
    
    hit_rate = hits / len(test_queries)
    print(f"Cache hit rate: {hit_rate:.1%}")
    assert hit_rate > 0.31, "Below target: 31%"
```

### 4. Embedding Generation
```python
async def benchmark_embeddings():
    """Test embedding generation speed"""
    chunks = generate_test_chunks(100)
    
    start = time.time()
    embeddings = await embedding_provider.embed_batch(chunks)
    duration = time.time() - start
    
    time_per_chunk = (duration * 1000) / len(chunks)
    print(f"Embedding time: {time_per_chunk:.1f}ms/chunk")
```

### 5. Redis Performance
```python
async def benchmark_redis():
    """Test Redis vector operations"""
    # Connection pool efficiency
    pool_stats = redis_client.connection_pool.get_stats()
    efficiency = pool_stats["hits"] / (pool_stats["hits"] + pool_stats["misses"])
    print(f"Connection pool efficiency: {efficiency:.1%}")
    
    # Memory usage
    info = await redis_client.info("memory")
    memory_mb = info["used_memory"] / (1024 * 1024)
    print(f"Redis memory usage: {memory_mb:.1f}MB")
    
    # Pipeline performance
    pipe = redis_client.pipeline()
    for i in range(1000):
        pipe.get(f"key:{i}")
    
    start = time.time()
    await pipe.execute()
    ops_per_sec = 1000 / (time.time() - start)
    print(f"Pipeline ops/sec: {ops_per_sec:.0f}")
```

## Benchmark Execution

### Quick Performance Check (1-2 minutes)
```bash
# Core metrics only
python -m pytest tests/benchmarks/test_performance.py -k "critical"
```

### Full Performance Suite (5-10 minutes)
```bash
# All performance tests
python -m pytest tests/benchmarks/ --benchmark-only \
  --benchmark-min-rounds=10 \
  --benchmark-disable-gc \
  --benchmark-warmup=on
```

### Component-Specific Tests
```bash
# Indexing only
/quality:performance-check --component=indexing

# Search performance
/quality:performance-check --component=search

# Cache effectiveness
/quality:performance-check --component=cache
```

## Performance Report

### Console Output
```
RAG Performance Report
======================
Document Indexing:    15.3 docs/sec  ✅ (target: >10)
Vector Search P95:    67ms          ✅ (target: <100ms)
Cache Hit Rate:       38.5%         ✅ (target: >31%)
Embedding Speed:      42ms/chunk    ✅ (target: <50ms)
Redis Memory:         1.3GB         ✅ (target: <2GB)
Connection Pool:      94% efficient ✅ (target: >90%)

Overall: ALL TARGETS MET
```

### Detailed Metrics
```json
{
  "indexing": {
    "docs_per_second": 15.3,
    "total_docs": 1000,
    "duration_seconds": 65.4
  },
  "search": {
    "p50_ms": 23,
    "p95_ms": 67,
    "p99_ms": 89,
    "queries_tested": 1000
  },
  "cache": {
    "hit_rate": 0.385,
    "total_queries": 1000,
    "hits": 385,
    "misses": 615
  }
}
```

## Load Testing

### Concurrent Operations
```python
async def load_test():
    """Test under concurrent load"""
    tasks = []
    
    # Simulate concurrent users
    for _ in range(10):
        tasks.append(process_documents())
        tasks.append(search_queries())
        tasks.append(cache_operations())
    
    start = time.time()
    await asyncio.gather(*tasks)
    duration = time.time() - start
    
    print(f"Load test completed in {duration:.1f}s")
```

### Stress Testing
```bash
# High load test
locust -f tests/load/locustfile.py \
  --users 100 \
  --spawn-rate 10 \
  --time 5m
```

## Optimization Recommendations

Based on performance results:

### If Indexing is Slow
- Increase batch size
- Use pipeline operations
- Optimize chunking strategy
- Enable async processing

### If Search is Slow
- Review index configuration
- Reduce result set size
- Implement caching layer
- Optimize embedding dimensions

### If Cache Hit Rate is Low
- Tune similarity threshold
- Implement query normalization
- Increase cache size
- Adjust TTL values

## Integration with PRP

Performance checks are run:
1. During `/prp:validate --performance`
2. After optimization phases
3. Before production deployment
4. As part of load testing

This performance check ensures RAG operations meet or exceed all performance targets.