# Redis Context Rules

## Patterns to Follow

- Always use connection pooling
- Pipeline for batch operations
- Implement retry logic
- Use transactions for atomic operations
- Monitor connection health

## Performance Focus

- Monitor memory usage
- Optimize key patterns
- Track operation latency
- Use appropriate data structures
- Implement circuit breakers

## Vector Operations

- Use Redis Vector Similarity Search
- Choose appropriate index algorithms
- Optimize vector dimensions
- Batch vector operations
- Monitor index performance

## Connection Management

```python
# Always use connection pooling
from redis.asyncio import ConnectionPool

pool = ConnectionPool(
    max_connections=50,
    decode_responses=True
)
```

## Error Handling

- Implement exponential backoff
- Use circuit breakers for failures
- Log all errors with context
- Provide graceful degradation
- Monitor error rates
