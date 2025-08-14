# EOL RAG Context MCP Server - Testing Guide

## Overview

The EOL RAG Context MCP Server includes comprehensive testing at multiple levels:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions with Redis
- **MCP Server Tests**: Test the MCP protocol implementation
- **End-to-End Tests**: Test complete workflows

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── test_document_processor.py   # Unit tests for document processing
├── test_indexer.py          # Unit tests for indexing
├── test_integration.py      # Integration tests with Redis
└── test_mcp_server.py       # MCP server functionality tests
```

## Running Tests

### Quick Start

```bash
# Run all unit tests (no Redis required)
./run_tests.sh

# Run tests with Redis integration
./run_tests.sh --redis

# Run all tests including integration
./run_tests.sh --all
```

### Manual Test Execution

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run specific test file
pytest tests/test_document_processor.py -v

# Run with coverage
pytest --cov=eol.rag_context --cov-report=html

# Run only marked tests
pytest -m "not redis"  # Skip Redis tests
pytest -m redis --redis  # Only Redis tests
```

## Redis Setup for Testing

### Using Docker (Recommended)

```bash
# Start Redis Stack with vector search
docker-compose -f docker-compose.test.yml up -d

# Verify Redis is running
redis-cli ping

# Run tests
pytest tests/test_integration.py --redis

# Stop Redis
docker-compose -f docker-compose.test.yml down
```

### Using Local Redis

Ensure you have Redis Stack (with RediSearch module) installed:

```bash
# macOS
brew tap redis-stack/redis-stack
brew install redis-stack

# Start Redis
redis-stack-server

# Run tests
pytest --redis
```

## Test Categories

### 1. Unit Tests

Test individual components without external dependencies:

- **Document Processor** (`test_document_processor.py`)
  - File type detection
  - Content chunking strategies
  - Metadata extraction
  - Language detection for code

- **Indexer** (`test_indexer.py`)
  - Folder scanning
  - File filtering (.gitignore support)
  - Source ID generation
  - Metadata tracking

- **MCP Server** (`test_mcp_server.py`)
  - Tool registration
  - Resource endpoints
  - Prompt templates
  - Request/response handling

### 2. Integration Tests

Test with real Redis connection:

- **Redis Operations** (`test_integration.py::TestRedisIntegration`)
  - Connection handling
  - Index creation
  - Document storage and retrieval
  - Hierarchical search

- **Semantic Cache** (`test_integration.py::TestSemanticCacheIntegration`)
  - Cache initialization
  - Similarity-based retrieval
  - Threshold tuning
  - Cache optimization

- **Knowledge Graph** (`test_integration.py::TestKnowledgeGraphIntegration`)
  - Entity extraction
  - Relationship building
  - Subgraph queries
  - Pattern discovery

- **File Watcher** (`test_integration.py::TestFileWatcherIntegration`)
  - Directory monitoring
  - Change detection
  - Real-time reindexing

### 3. End-to-End Tests

Complete workflow testing:

```python
# test_integration.py::TestEndToEndIntegration
- Full indexing pipeline
- Knowledge graph construction
- Search and retrieval
- Cache integration
```

## MCP Server Testing

### Testing MCP Tools

```python
# Example: Testing index_directory tool
async def test_index_directory_tool(server):
    request = IndexDirectoryRequest(
        path="/test/path",
        recursive=True,
        watch=False
    )

    result = await index_tool.function(request, Mock())
    assert result["source_id"] == "test_source"
```

### Testing MCP Resources

```python
# Example: Testing context retrieval
async def test_get_context_resource(server):
    result = await context_resource.function("test query")
    assert result["cached"] is False
    assert "context" in result
```

### Testing MCP Prompts

```python
# Example: Testing structured query prompt
async def test_structured_query_prompt(server):
    result = await query_prompt.function()
    assert "Main Intent" in result
```

## Advanced Testing Scenarios

### 1. Performance Testing

```python
# Load test with many documents
async def test_bulk_indexing_performance():
    start = time.time()
    for i in range(1000):
        await indexer.index_file(f"file_{i}.txt")
    duration = time.time() - start
    assert duration < 60  # Should index 1000 files in < 60s
```

### 2. Concurrent Access Testing

```python
# Test concurrent indexing and searching
async def test_concurrent_operations():
    tasks = [
        indexer.index_file("file1.txt"),
        indexer.index_file("file2.txt"),
        redis_store.vector_search(query_embedding)
    ]
    results = await asyncio.gather(*tasks)
    assert all(r is not None for r in results)
```

### 3. Error Recovery Testing

```python
# Test Redis connection failure handling
async def test_redis_reconnection():
    # Simulate connection loss
    await redis_store.close()

    # Should reconnect automatically
    await redis_store.connect_async()
    assert await redis_store.async_redis.ping()
```

## Testing Best Practices

### 1. Use Fixtures for Setup

```python
@pytest.fixture
async def indexed_corpus(redis_store, sample_documents):
    """Pre-index documents for tests."""
    # Index documents
    # Return indexer
```

### 2. Mock External Dependencies

```python
@pytest.fixture
def mock_llm():
    """Mock LLM for embedding generation."""
    return Mock(embed=AsyncMock(return_value=np.random.randn(128)))
```

### 3. Clean Up After Tests

```python
@pytest.fixture
async def redis_store():
    store = RedisVectorStore(config)
    yield store
    await store.async_redis.flushdb()  # Clean test data
    await store.close()
```

### 4. Test Edge Cases

- Empty documents
- Large files (> max_size)
- Invalid file formats
- Concurrent modifications
- Network failures

## Continuous Integration

### GitHub Actions Setup

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      redis:
        image: redis/redis-stack:latest
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run tests
        run: |
          pytest --redis --cov=eol.rag_context

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Debugging Tests

### Enable Logging

```python
# In conftest.py or test file
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Use pytest debugging

```bash
# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s

# Verbose output
pytest -vv
```

### Inspect Redis Data

```bash
# Connect to test Redis
redis-cli -n 15  # Test DB

# List all keys
KEYS *

# Inspect vector index
FT.INFO eol_context_chunk

# Search vectors
FT.SEARCH eol_context_chunk "*" LIMIT 0 10
```

## Test Coverage

### Generate Coverage Report

```bash
# Run with coverage
pytest --cov=eol.rag_context --cov-report=html

# View report
open htmlcov/index.html
```

### Coverage Goals

- Unit tests: > 80% coverage
- Integration tests: > 60% coverage
- Critical paths: 100% coverage

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**

   ```bash
   # Check Redis is running
   redis-cli ping

   # Check correct port
   lsof -i :6379
   ```

2. **Import Errors**

   ```bash
   # Ensure package is installed
   pip install -e .
   ```

3. **Slow Tests**

   ```bash
   # Run tests in parallel
   pytest -n auto
   ```

4. **Flaky Tests**

   ```bash
   # Rerun failures
   pytest --reruns 3
   ```

## Additional Testing Tools

### MCP Client Testing

Test the MCP server with the official MCP client:

```bash
# Install MCP client
pip install mcp-client

# Test server
mcp-client test eol-rag-context
```

### Load Testing

```python
# Use locust for load testing
from locust import HttpUser, task

class RAGUser(HttpUser):
    @task
    def search_context(self):
        self.client.post("/search", json={"query": "test"})
```

### Benchmarking

```python
# Benchmark indexing speed
import timeit

def benchmark_indexing():
    result = timeit.timeit(
        "indexer.index_file('test.txt')",
        number=100
    )
    print(f"Average time: {result/100:.3f}s")
```

## Contributing Tests

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Add integration tests for Redis interactions
4. Document test scenarios
5. Update this guide if needed

## Support

For test-related issues:

- Check the [GitHub Issues](https://github.com/eoln/eol/issues)
- Review test output carefully
- Enable debug logging
- Isolate failing tests
