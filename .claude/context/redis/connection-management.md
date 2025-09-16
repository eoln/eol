# Redis Connection Management

## Connection Pool Configuration

### Basic Pool Setup

```python
from redis.asyncio import Redis, ConnectionPool
from redis.retry import Retry
from redis.backoff import ExponentialBackoff

# Configure retry strategy
retry = Retry(
    backoff=ExponentialBackoff(),
    retries=3
)

# Create connection pool
pool = ConnectionPool(
    host="localhost",
    port=6379,
    db=0,
    max_connections=50,
    decode_responses=True,
    retry=retry,
    retry_on_error=[ConnectionError, TimeoutError]
)

# Create client
redis_client = Redis(connection_pool=pool)
```

## Health Checking

### Connection Health Monitor

```python
async def check_redis_health() -> Dict[str, Any]:
    """Monitor Redis connection health"""
    try:
        # Ping test
        start = time.time()
        await redis_client.ping()
        latency = (time.time() - start) * 1000

        # Get pool stats
        pool_stats = {
            "created_connections": pool.created_connections,
            "available_connections": len(pool._available_connections),
            "in_use_connections": len(pool._in_use_connections)
        }

        return {
            "status": "healthy",
            "latency_ms": latency,
            "pool_stats": pool_stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
```

## Circuit Breaker Pattern

```python
from circuitbreaker import circuit

class RedisCircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30):
        self.breaker = circuit(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=RedisError
        )

    @circuit
    async def execute(self, operation, *args, **kwargs):
        """Execute Redis operation with circuit breaker"""
        return await operation(*args, **kwargs)
```

## Connection Lifecycle

### Graceful Shutdown

```python
async def shutdown_redis():
    """Gracefully close Redis connections"""
    try:
        # Flush pending operations
        await redis_client.flushdb()

        # Close all connections
        await redis_client.close()
        await pool.disconnect()

        logger.info("Redis connections closed successfully")
    except Exception as e:
        logger.error(f"Error during Redis shutdown: {e}")
```

### Connection Recovery

```python
async def ensure_connection():
    """Ensure Redis connection is alive"""
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            await redis_client.ping()
            return True
        except (ConnectionError, TimeoutError):
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))
            else:
                raise ConnectionError("Failed to connect to Redis")
```

## Best Practices

1. Always use connection pooling
2. Implement health checks
3. Use circuit breakers for resilience
4. Monitor pool metrics
5. Handle connection errors gracefully
6. Set appropriate timeouts
7. Clean up connections on shutdown
