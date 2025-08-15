# Redis Performance Tuning

## Memory Optimization

### Key Design Patterns

```python
# Good: Hierarchical, predictable
key = "rag:doc:12345:chunk:1"

# Bad: Long, redundant
key = "retrieval_augmented_generation_document_12345_chunk_number_1"
```

### Memory Policies

```redis
# Set max memory
CONFIG SET maxmemory 2gb

# Set eviction policy
CONFIG SET maxmemory-policy allkeys-lru
```

## Query Performance

### Pipeline Operations

```python
async def bulk_get(keys: List[str]) -> List[Any]:
    """Use pipeline for multiple operations"""
    pipe = redis_client.pipeline()
    for key in keys:
        pipe.get(key)
    return await pipe.execute()
```

### Connection Pooling

```python
# Optimal pool configuration
pool = ConnectionPool(
    max_connections=50,
    max_idle_time=30,
    retry_on_timeout=True,
    socket_keepalive=True,
    socket_keepalive_options={
        1: 1,  # TCP_KEEPIDLE
        2: 2,  # TCP_KEEPINTVL
        3: 3,  # TCP_KEEPCNT
    }
)
```

## Monitoring Metrics

### Key Performance Indicators

```python
async def get_redis_metrics():
    info = await redis_client.info()
    return {
        "used_memory": info["used_memory_human"],
        "connected_clients": info["connected_clients"],
        "ops_per_sec": info["instantaneous_ops_per_sec"],
        "hit_rate": info["keyspace_hits"] /
                   (info["keyspace_hits"] + info["keyspace_misses"]),
        "evicted_keys": info["evicted_keys"]
    }
```

### Performance Benchmarks

- Vector search: <100ms for 10k documents
- Bulk writes: >1000 docs/sec
- Cache lookup: <10ms
- Connection pool efficiency: >90%

## Optimization Checklist

- [ ] Use appropriate data structures
- [ ] Implement connection pooling
- [ ] Batch operations with pipeline
- [ ] Monitor memory usage
- [ ] Set appropriate TTLs
- [ ] Use compression for large values
- [ ] Index only necessary fields
