# Integration Tests

This directory contains integration tests for the EOL RAG Context MCP server.

## Important: Real Redis v8 Required

All integration tests in this directory run against **real Redis v8 instances**, not mocks. This ensures that:
- Vector search operations work correctly with Redis Stack
- HNSW indexing performs as expected
- All tutorial code examples are validated against actual Redis
- Performance metrics are realistic

## Test Files

### `test_tutorial_examples.py`
Tests all code examples from the TUTORIAL.md to ensure they work with real Redis v8:
- Server initialization with Redis fixtures
- Directory and file indexing with actual storage
- Search operations using Redis vector search
- Knowledge graph queries with real graph storage
- Semantic caching with Redis persistence
- Context window management
- All example code from documentation

### Key Features
- **Real Redis Connections**: Uses `redis_store` fixture from `conftest.py`
- **Actual Data Storage**: Documents are stored in Redis during tests
- **Vector Operations**: Real embeddings and similarity search
- **No Mocks**: Direct interaction with Redis v8 features

## Running Integration Tests

### Prerequisites
1. **Redis Stack v8** must be running:
   ```bash
   docker run -d -p 6379:6379 redis/redis-stack:latest
   ```

2. **Python dependencies** installed:
   ```bash
   pip install -e .
   ```

### Run Tests

#### Automated (Recommended)
```bash
# Automatically starts Redis, runs tests, and stops Redis
./test_all.sh
```

#### Manual
```bash
# Start Redis
docker run -d -p 6379:6379 redis/redis-stack:latest

# Run integration tests
pytest tests/integration/ -xvs

# Or specific test file
pytest tests/integration/test_tutorial_examples.py -xvs
```

### Verify Redis Connection
```bash
python verify_redis_integration.py
```

This script will:
- Check Redis connection
- Verify Redis version
- Test vector operations
- Confirm tutorial examples work

## Test Configuration

### Redis Fixtures (`conftest.py`)
- `redis_store`: Provides connected Redis client with indexes
- `indexer_instance`: Real indexer using Redis store
- `semantic_cache_instance`: Cache with Redis backend
- `knowledge_graph_instance`: Graph builder with Redis storage
- `file_watcher_instance`: File watcher with real indexer

### Environment Variables
- `REDIS_HOST`: Redis host (default: localhost)
- `REDIS_PORT`: Redis port (default: 6379)

## What's Tested

### Tutorial Examples
Each test corresponds to a code example in TUTORIAL.md:
1. **Basic Usage** - Server initialization
2. **Indexing** - Single files and directories
3. **Watching** - Real-time file monitoring
4. **Search** - Basic, hierarchical, and filtered
5. **Knowledge Graph** - Entity relationships
6. **Caching** - Semantic similarity caching
7. **Context Windows** - Token management
8. **Integration Examples** - Code assistant, doc search
9. **Best Practices** - Batch operations, optimization
10. **Health Checks** - Server status monitoring

### Real Operations
Unlike unit tests, these integration tests perform:
- Actual file indexing with content extraction
- Real vector embeddings and storage
- Genuine similarity searches
- Persistent caching operations
- Live file watching with events

## Troubleshooting

### Redis Connection Failed
```bash
# Check if Redis is running
docker ps | grep redis

# If not, start it
docker run -d -p 6379:6379 redis/redis-stack:latest
```

### Module Not Found
```bash
# Install in development mode
pip install -e .
```

### Tests Hanging
- Check Redis memory usage: `redis-cli INFO memory`
- Clear Redis data: `redis-cli FLUSHALL`
- Restart Redis container

## CI/CD Integration

GitHub Actions workflow runs these tests with:
```yaml
services:
  redis:
    image: redis/redis-stack:latest
    ports:
      - 6379:6379
```

## Notes

- Tests may be slower than unit tests due to real I/O
- Each test cleans up its data after completion
- Redis data persists between test runs unless cleared
- Use `pytest -x` to stop on first failure for debugging