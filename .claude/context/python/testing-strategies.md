# Testing Strategies with Pytest

## Test Structure

### Basic Test Organization
```python
# tests/test_document_processor.py
import pytest
from unittest.mock import Mock, AsyncMock, patch
from eol.rag_context import DocumentProcessor

class TestDocumentProcessor:
    """Group related tests in classes"""
    
    @pytest.fixture
    async def processor(self):
        """Setup processor for tests"""
        proc = DocumentProcessor()
        yield proc
        # Cleanup if needed
        await proc.close()
    
    async def test_process_text_document(self, processor):
        """Test processing text documents"""
        result = await processor.process("test.txt", "content")
        assert result.chunks
        assert len(result.chunks) > 0
```

## Fixtures

### Reusable Test Setup
```python
@pytest.fixture(scope="session")
async def redis_client():
    """Session-scoped Redis client"""
    client = await create_test_redis()
    yield client
    await client.flushdb()
    await client.close()

@pytest.fixture
async def sample_documents():
    """Provide sample documents for testing"""
    return [
        Document("doc1.txt", "Content 1"),
        Document("doc2.md", "# Header\nContent 2"),
        Document("doc3.py", "def func():\n    pass")
    ]

@pytest.fixture(autouse=True)
async def cleanup_temp_files():
    """Auto-cleanup temporary files after each test"""
    yield
    # Cleanup code
    for file in Path("/tmp/test_*").glob("*"):
        file.unlink()
```

## Mocking

### Mock External Dependencies
```python
@pytest.fixture
def mock_llm():
    """Mock LLM API calls"""
    with patch("eol.rag_context.llm.generate") as mock:
        mock.return_value = AsyncMock(
            return_value="Generated response"
        )
        yield mock

async def test_with_mocked_llm(mock_llm):
    """Test with mocked LLM"""
    result = await generate_with_context("query")
    mock_llm.assert_called_once()
    assert result == "Generated response"
```

### Mock Redis Operations
```python
@pytest.fixture
def mock_redis():
    """Mock Redis for unit tests"""
    mock = AsyncMock()
    mock.get.return_value = '{"key": "value"}'
    mock.set.return_value = True
    mock.search.return_value = [
        {"content": "result1", "score": 0.9},
        {"content": "result2", "score": 0.8}
    ]
    return mock
```

## Async Testing

### Test Async Functions
```python
@pytest.mark.asyncio
async def test_async_operation():
    """Test async function"""
    result = await async_function()
    assert result is not None

# Alternative: use pytest-asyncio auto mode
# pytest.ini or pyproject.toml:
# [tool.pytest.ini_options]
# asyncio_mode = "auto"
```

### Test Concurrent Operations
```python
async def test_concurrent_processing():
    """Test parallel processing"""
    docs = [Document(f"doc{i}") for i in range(10)]
    
    start = time.time()
    results = await process_documents_parallel(docs)
    duration = time.time() - start
    
    assert len(results) == 10
    assert duration < 2.0  # Should be faster than sequential
```

## Parametrized Tests

```python
@pytest.mark.parametrize("chunk_size,expected_chunks", [
    (100, 10),
    (200, 5),
    (500, 2),
])
async def test_chunking_strategies(chunk_size, expected_chunks):
    """Test different chunk sizes"""
    text = "x" * 1000
    chunks = chunk_text(text, chunk_size)
    assert len(chunks) == expected_chunks

@pytest.mark.parametrize("file_type,processor_class", [
    ("txt", TextProcessor),
    ("md", MarkdownProcessor),
    ("py", CodeProcessor),
])
async def test_file_processors(file_type, processor_class):
    """Test correct processor selection"""
    processor = get_processor(f"test.{file_type}")
    assert isinstance(processor, processor_class)
```

## Integration Tests

```python
@pytest.mark.integration
class TestRedisIntegration:
    """Integration tests requiring Redis"""
    
    @pytest.fixture(scope="class")
    async def redis_store(self):
        """Setup Redis for integration tests"""
        store = RedisVectorStore()
        await store.initialize()
        yield store
        await store.cleanup()
    
    async def test_end_to_end_indexing(self, redis_store):
        """Test complete indexing workflow"""
        # Index documents
        docs = await load_test_documents()
        await redis_store.index_documents(docs)
        
        # Search
        results = await redis_store.search("test query")
        assert len(results) > 0
        
        # Verify metadata
        assert all(r.metadata for r in results)
```

## Performance Tests

```python
@pytest.mark.benchmark
async def test_indexing_performance(benchmark):
    """Benchmark document indexing"""
    docs = generate_test_documents(100)
    
    async def index():
        indexer = DocumentIndexer()
        await indexer.index_batch(docs)
    
    result = benchmark(index)
    assert result.stats["mean"] < 10.0  # Should index in <10s
```

## Coverage Configuration

### pyproject.toml
```toml
[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/test_*.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
fail_under = 80
```

## Test Helpers

```python
# tests/helpers.py
async def create_test_document(
    content: str = "test content",
    metadata: Optional[Dict] = None
) -> Document:
    """Helper to create test documents"""
    return Document(
        content=content,
        metadata=metadata or {"test": True}
    )

async def assert_eventually(
    condition: Callable[[], bool],
    timeout: float = 5.0,
    interval: float = 0.1
):
    """Assert condition becomes true within timeout"""
    start = time.time()
    while time.time() - start < timeout:
        if condition():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"Condition not met within {timeout}s")
```

## Best Practices
1. Use descriptive test names
2. One assertion per test (when possible)
3. Use fixtures for setup/teardown
4. Mock external dependencies
5. Test both success and failure cases
6. Use parametrize for similar tests
7. Mark slow tests appropriately
8. Maintain >80% coverage