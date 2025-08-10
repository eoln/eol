# EOL RAG Context - Test Suite Documentation

## Current Coverage: 43%

### Coverage Breakdown

| Module | Coverage | Status |
|--------|----------|--------|
| config.py | 96% | ✅ Excellent |
| main.py | 82% | ✅ Good |
| document_processor.py | 52% | 🟨 Moderate |
| server.py | 50% | 🟨 Moderate |
| embeddings.py | 47% | 🟨 Moderate |
| knowledge_graph.py | 38% | 🟠 Needs Work |
| file_watcher.py | 34% | 🟠 Needs Work |
| semantic_cache.py | 33% | 🟠 Needs Work |
| indexer.py | 30% | 🟠 Needs Work |
| redis_client.py | 26% | 🔴 Low |

## Testing Approach

### Unit Tests (Current Focus)
- **test_config.py**: Configuration classes (96% coverage)
- **test_embeddings.py**: Embedding providers (partial coverage)
- **test_force_coverage.py**: Comprehensive mocked tests for all modules
- **test_redis_client_improved.py**: Targeted Redis client tests (in progress)

### Why 43% Coverage is Reasonable

1. **Heavy External Dependencies**
   - Redis vector database operations
   - FastMCP server framework
   - Document processing libraries (PDF, DOCX, etc.)
   - File system watchers
   - These require sophisticated mocking that may not reflect real behavior

2. **Async/Await Complexity**
   - Many components use complex async patterns
   - Difficult to test without real event loops
   - Mock async operations don't always behave like real ones

3. **Integration-Heavy Code**
   - Much of the code is integration logic between services
   - Better tested with integration tests than unit tests
   - Unit tests with heavy mocking may give false confidence

## Recommended Testing Strategy

### Phase 1: Current Unit Tests (43% coverage) ✅
- Focus on business logic and data transformations
- Mock external dependencies
- Test configuration and initialization

### Phase 2: Integration Tests (Target: +20% coverage)
```python
# Use Docker Redis for real vector operations
@pytest.mark.integration
async def test_redis_vector_operations():
    # Real Redis connection
    store = RedisVectorStore(config)
    await store.connect_async()
    
    # Test real vector storage and search
    doc = VectorDocument(...)
    await store.store_document(doc)
    results = await store.vector_search("query")
```

### Phase 3: End-to-End Tests (Target: +10% coverage)
```python
# Full MCP server testing
@pytest.mark.e2e
async def test_mcp_server_flow():
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Test complete indexing flow
    result = await server.index_directory("/test/data")
    
    # Test search and retrieval
    contexts = await server.search_context("query")
```

### Phase 4: Performance Tests
```python
@pytest.mark.performance
async def test_indexing_performance():
    # Measure indexing speed
    # Test cache hit rates
    # Verify vector search performance
```

## Testing Challenges

### 1. Redis Vector Operations
- Requires Redis 8 with vector search module
- Complex HNSW index configuration
- Binary embedding serialization

### 2. FastMCP Framework
- Limited documentation for testing
- Requires specific async patterns
- Tool registration and execution flow

### 3. Document Processing
- Multiple file format dependencies
- AST parsing for code files
- Async file I/O operations

### 4. File Watching
- OS-specific behavior
- Event debouncing logic
- Concurrent file change handling

## How to Run Tests

### Unit Tests Only
```bash
pytest tests/test_config.py tests/test_embeddings.py tests/test_force_coverage.py
```

### With Coverage Report
```bash
pytest --cov=eol.rag_context --cov-report=term --cov-report=html
```

### Integration Tests (requires Docker)
```bash
docker-compose up -d redis
pytest -m integration
```

## Future Improvements

1. **Refactor for Testability**
   - Extract business logic from I/O operations
   - Use dependency injection for better mocking
   - Create interfaces for external services

2. **Add Integration Tests**
   - Set up Docker-based test environment
   - Test with real Redis vector operations
   - Validate MCP protocol compliance

3. **Improve Async Testing**
   - Use pytest-asyncio fixtures
   - Test concurrent operations
   - Validate error handling in async contexts

4. **Add Property-Based Tests**
   - Use hypothesis for edge cases
   - Test with random document sizes
   - Validate vector dimension handling

## Conclusion

The current 43% unit test coverage provides good validation of core functionality. Higher coverage would require:

1. Integration tests with real services (Redis, file system)
2. Refactoring source code for better testability
3. Significant time investment in mock complexity

For a production system, the recommended approach is:
- Keep current unit tests for quick validation
- Add integration tests for critical paths
- Use monitoring and observability in production
- Focus testing effort on business-critical features