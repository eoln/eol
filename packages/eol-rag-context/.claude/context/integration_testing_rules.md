# Integration Testing Rules for EOL RAG Context

## Core Principles

### 1. No Mocking of Integration Contracts
- **NEVER mock Redis connections** in integration tests
- **NEVER mock vector operations** that are part of the integration contract
- **NEVER create fallback tests** that hide real failures
- If a dependency is missing, the test should **fail or skip properly**, not pretend to pass

### 2. Real Dependencies Required
Integration tests MUST use real instances of:
- Redis Stack Server with RediSearch module (not regular Redis)
- Actual numpy arrays for embeddings (not Mock objects)
- Real file system operations
- Actual async event loops

## Setting Up Integration Tests

### 1. Install Dependencies Systematically
```bash
# Use the provided setup script
./setup_test_environment.sh

# Or use Make commands
make install-dev
make redis-start
```

### 2. Redis Stack Configuration
- **MUST use Redis Stack**, not regular Redis
- Required modules: RediSearch (FT.* commands)
- Test for module availability:
```python
try:
    store.create_hierarchical_indexes(embedding_dim=384)
except Exception as e:
    if "unknown command" in str(e).lower() and "ft.create" in str(e).lower():
        pytest.skip(f"RediSearch module not available: {e}")
```

### 3. Embedding Dimensions
- Use **384 dimensions** for all-MiniLM-L6-v2 model
- Never use 768 (that's for larger models)
- Always return real numpy arrays:
```python
class MockSentenceTransformer:
    def encode(self, texts):
        return [np.random.randn(384).astype(np.float32) for _ in texts]
```

## Common Integration Test Patterns

### 1. Fixture Scope Management
```python
# Use function scope to avoid event loop conflicts
@pytest.fixture(scope="function")  # NOT "session"
async def redis_store(redis_config):
    store = RedisVectorStore(redis_config, IndexConfig())
    await store.connect_async()
    store.connect()
    yield store
    await store.close()
```

### 2. Key Format Conventions
Redis uses hierarchical prefixes for different document levels:
- Level 1 (concepts): `concept:{doc_id}`
- Level 2 (sections): `section:{doc_id}`
- Level 3 (chunks): `chunk:{doc_id}`

Never use generic prefixes like `doc:{level}:{id}`

### 3. Method Return Types
The `vector_search` method returns tuples, not dictionaries:
```python
# Returns: List[Tuple[str, float, Dict[str, Any]]]
# Format: [(doc_id, score, data), ...]
results = await redis_store.vector_search(query_embedding, k=5)
for doc_id, score, data in results:
    # Process each result
```

### 4. API Method Names
Be aware of actual method names in the codebase:
- `DocumentIndexer.index_folder()` (NOT `index_directory`)
- `DocumentIndexer.index_file()` (for single files)
- `KnowledgeGraphBuilder.query_subgraph()` (NOT `query_entity`)
- `SemanticCache.get_stats()` (synchronous, not async)

## Known Limitations

### 1. Redis TAG Field Filtering
TAG field filtering with KNN queries is not currently supported in Redis Stack.
```python
# This will fail:
results = await store.vector_search(
    query_embedding, 
    filters={"parent": "some_id"}  # TAG field filter with KNN
)

# Workaround: Get more results and filter manually
results = await store.vector_search(query_embedding, k=20)
filtered = [r for r in results if r[2].get("parent") == "some_id"]
```

### 2. File Watcher Testing
File watcher tests may need to be skipped if watchdog is not available:
```python
if file_watcher_instance is None:
    pytest.skip("File watcher not available for testing")
```

## Running Integration Tests

### 1. Individual Test Files
```bash
# Run specific integration test file
source .venv/bin/activate
pytest tests/integration/test_redis_integration.py -v

# Run with specific markers
pytest tests/integration -m integration -v
```

### 2. Debugging Failed Tests
```bash
# Run with detailed output
pytest tests/integration/test_file.py::TestClass::test_method -xvs

# Show only failures with traceback
pytest tests/integration -x --tb=short
```

### 3. Check Dependencies
Before running tests, ensure:
```bash
# Redis Stack is running
redis-cli ping

# RediSearch module is loaded
redis-cli MODULE LIST | grep search

# Python dependencies are installed
pip list | grep redis
```

## Troubleshooting Integration Tests

### Problem: "unknown command 'FT.CREATE'"
**Solution:** Install Redis Stack, not regular Redis
```bash
brew install --cask redis-stack-server
redis-stack-server --daemonize yes
```

### Problem: "attached to a different loop"
**Solution:** Use function-scoped fixtures, not session-scoped

### Problem: Mock object has no attribute 'dtype'
**Solution:** Return real numpy arrays, not Mock objects

### Problem: Dimension mismatch errors
**Solution:** Use 384 dimensions for all-MiniLM-L6-v2

### Problem: "Syntax error at offset X near Y"
**Solution:** TAG field filtering with KNN not supported, use workaround

## Best Practices

1. **Always test with real Redis Stack** - no in-memory substitutes
2. **Use real embeddings** - even if they're random, they must be numpy arrays
3. **Check method existence** - verify the actual API before writing tests
4. **Handle missing dependencies gracefully** - skip, don't fake
5. **Test the actual behavior** - not what you think it should be
6. **Document known limitations** - add TODOs for issues that need fixing

## Example Integration Test

```python
@pytest.mark.integration
async def test_redis_vector_operations(redis_store, embedding_manager):
    """Test real Redis vector operations."""
    # Create document with real embedding
    content = "Test document"
    embedding = await embedding_manager.get_embedding(content)
    
    doc = VectorDocument(
        id="test_1",
        content=content,
        embedding=embedding,  # Real numpy array
        hierarchy_level=1
    )
    
    # Store in real Redis
    await redis_store.store_document(doc)
    
    # Verify with correct key format
    key = f"concept:{doc.id}"  # Correct prefix for level 1
    data = await redis_store.async_redis.hgetall(key)
    assert data is not None
    
    # Search returns tuples
    results = await redis_store.vector_search(embedding, k=5)
    for doc_id, score, data in results:
        assert isinstance(doc_id, str)
        assert isinstance(score, float)
        assert isinstance(data, dict)
```

## Summary

Integration tests must test **real integration**, not mocked behavior. When writing or fixing integration tests:

1. Use real Redis Stack with RediSearch
2. Return real numpy arrays for embeddings
3. Use correct method names and signatures
4. Handle missing dependencies by skipping, not faking
5. Expect and handle known limitations
6. Never hide failures with fallbacks

Remember: **"If it's failing, it means integration is failing"** - don't add fallbacks that make tests pass when they shouldn't.