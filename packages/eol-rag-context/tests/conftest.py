"""Pytest configuration and fixtures for unit tests."""

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from eol.rag_context.config import RAGConfig, RedisConfig
from eol.rag_context.document_processor import DocumentProcessor
from eol.rag_context.indexer import DocumentIndexer


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def test_config(temp_dir: Path) -> RAGConfig:
    """Create test configuration."""
    config = RAGConfig()
    config.data_dir = temp_dir / "data"
    config.index_dir = temp_dir / "indexes"
    config.data_dir.mkdir(exist_ok=True)
    config.index_dir.mkdir(exist_ok=True)

    # Use smaller embedding dimension for tests
    config.embedding.dimension = 384  # Match all-MiniLM-L6-v2
    config.embedding.model_name = "all-MiniLM-L6-v2"

    # Disable cache for tests
    config.cache.enabled = False

    return config


@pytest.fixture
def redis_config() -> RedisConfig:
    """Redis configuration for tests."""
    import os

    redis_port = int(os.environ.get("REDIS_PORT", "6379"))
    return RedisConfig(host="localhost", port=redis_port, db=15)  # Use separate DB for tests


@pytest.fixture
def redis_store() -> Mock:
    """Create mock Redis store for unit tests."""

    # Use a class to hold the state
    class MockRedisState:
        def __init__(self):
            self.stored_data = {}

    state = MockRedisState()

    def mock_hset(key, mapping=None, **kwargs):
        """Mock hset to store data with byte keys."""
        if mapping:
            state.stored_data[key] = {
                k.encode() if isinstance(k, str) else k: (
                    str(v).encode() if not isinstance(v, bytes) else v
                )
                for k, v in mapping.items()
            }
        return 1

    def mock_hgetall(key):
        """Mock hgetall to return stored data."""
        result = state.stored_data.get(key, {})
        return result

    def mock_delete(*keys):
        """Mock delete to remove keys."""
        deleted = 0
        for key in keys:
            if key in state.stored_data:
                del state.stored_data[key]
                deleted += 1
        return deleted

    def mock_keys(pattern):
        """Mock keys to return matching keys."""
        prefix = pattern.replace("*", "")
        matching = [k for k in state.stored_data.keys() if k.startswith(prefix)]
        # Always return byte-encoded keys
        return [k.encode() if isinstance(k, str) else k for k in matching]

    def mock_scan(cursor=0, match=None, count=100):
        """Mock scan to return matching keys."""
        if match:
            prefix = match.replace("*", "")
            matching_keys = [k for k in state.stored_data.keys() if k.startswith(prefix)]
        else:
            matching_keys = list(state.stored_data.keys())
        # Always return byte-encoded keys in scan results
        encoded_keys = [k.encode() if isinstance(k, str) else k for k in matching_keys]
        return (0, encoded_keys)  # Return cursor 0 to indicate end

    store = Mock()
    store.connect = Mock()
    store.connect_async = AsyncMock()
    store.close = AsyncMock()

    # Create async_redis mock
    store.async_redis = AsyncMock()
    store.async_redis.flushdb = AsyncMock()
    store.async_redis.hgetall = AsyncMock(side_effect=mock_hgetall)
    store.async_redis.hset = AsyncMock(side_effect=mock_hset)
    store.async_redis.hget = AsyncMock(return_value=None)
    store.async_redis.delete = AsyncMock(side_effect=mock_delete)
    store.async_redis.keys = AsyncMock(side_effect=mock_keys)
    store.async_redis.ft = Mock(return_value=AsyncMock())
    store.async_redis.expire = AsyncMock()
    store.async_redis.scan = AsyncMock(side_effect=mock_scan)
    store.async_redis.hincrby = AsyncMock()

    # Create sync redis mock
    # The RedisVectorStore has self.redis = Redis client
    # The indexer accesses self.redis.redis where self.redis is RedisVectorStore
    # So we need store.redis to be the Redis client mock
    redis_client_mock = Mock()
    redis_client_mock.hgetall = Mock(side_effect=mock_hgetall)
    redis_client_mock.hset = Mock(side_effect=mock_hset)
    redis_client_mock.hget = Mock(return_value=None)
    redis_client_mock.delete = Mock(side_effect=mock_delete)
    redis_client_mock.keys = Mock(side_effect=mock_keys)
    redis_client_mock.ping = Mock(return_value=True)
    redis_client_mock.expire = Mock()
    redis_client_mock.scan = Mock(side_effect=mock_scan)
    redis_client_mock.hincrby = Mock()

    store.redis = redis_client_mock

    # Store methods
    store.store_document = AsyncMock()
    store.vector_search = AsyncMock(return_value=[])
    store.hierarchical_search = AsyncMock(return_value=[])
    store.get_document_tree = AsyncMock(return_value={})
    store.create_hierarchical_indexes = Mock()

    return store


@pytest.fixture
def mock_embedding_manager(test_config: RAGConfig) -> Mock:
    """Create mock embedding manager."""
    manager = Mock()
    manager.get_embedding = AsyncMock(return_value=np.random.randn(384).astype(np.float32))
    manager.get_embeddings = AsyncMock(return_value=np.random.randn(10, 384).astype(np.float32))
    manager.clear_cache = Mock()
    manager.get_cache_stats = Mock(return_value={"hits": 0, "misses": 0})
    return manager


@pytest.fixture
def sample_documents(temp_dir: Path) -> dict[str, Path]:
    """Create sample documents for testing."""
    docs = {}

    # Markdown file
    md_file = temp_dir / "test.md"
    md_file.write_text(
        """# Test Document

## Introduction
This is a test document for RAG indexing.

## Code Example
```python
def hello_world():
    print("Hello, World!")
```

## Conclusion
This concludes our test document.
"""
    )
    docs["markdown"] = md_file

    # Python file
    py_file = temp_dir / "test.py"
    py_file.write_text(
        """
def factorial(n):
    \"\"\"Calculate factorial of n.\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n - 1)

class TestClass:
    def __init__(self):
        self.value = 42
"""
    )
    docs["python"] = py_file

    # JSON file
    json_file = temp_dir / "test.json"
    json_file.write_text(
        """
{
    "name": "Test Project",
    "version": "1.0.0",
    "dependencies": {
        "numpy": "^1.24.0",
        "pandas": "^2.0.0"
    }
}
"""
    )
    docs["json"] = json_file

    # JavaScript file
    js_file = temp_dir / "test.js"
    js_file.write_text(
        """
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

export default fibonacci;
"""
    )
    docs["javascript"] = js_file

    # Plain text file
    txt_file = temp_dir / "test.txt"
    txt_file.write_text(
        """This is a plain text file.
It contains multiple lines.
Used for testing text processing."""
    )
    docs["text"] = txt_file

    return docs


@pytest.fixture
async def indexed_documents(
    redis_store, mock_embedding_manager, test_config, sample_documents
) -> "DocumentIndexer":
    """Create indexer with pre-indexed documents."""
    from eol.rag_context.indexer import DocumentIndexer

    processor = DocumentProcessor(test_config.document, test_config.chunking)
    indexer = DocumentIndexer(test_config, processor, mock_embedding_manager, redis_store)

    # Mock some indexed documents
    indexer.stats = {
        "documents_indexed": len(sample_documents),
        "chunks_created": len(sample_documents) * 5,
        "concepts_extracted": len(sample_documents) * 2,
        "sections_created": len(sample_documents) * 3,
        "errors": 0,
    }

    return indexer


@pytest.fixture
def scanner(test_config):
    """Create folder scanner."""
    from eol.rag_context.indexer import FolderScanner

    return FolderScanner(test_config)


@pytest.fixture
async def server(test_config):
    """Create MCP server instance with mocks."""
    from eol.rag_context.server import EOLRAGContextServer

    server = EOLRAGContextServer(test_config)

    # Mock components to avoid Redis dependency
    server.redis_store = AsyncMock()
    server.embedding_manager = AsyncMock()
    server.document_processor = Mock()
    server.indexer = AsyncMock()
    server.semantic_cache = AsyncMock()
    server.knowledge_graph = AsyncMock()
    server.file_watcher = AsyncMock()

    # Don't re-register as it won't work with mocked components
    # The registration happens in __init__ but references None components

    # Mock methods
    server.indexer.index_folder = AsyncMock(
        return_value=Mock(
            source_id="test_source",
            path=Path("/test"),
            indexed_at=1234567890,
            file_count=10,
            total_chunks=50,
        )
    )

    server.indexer.list_sources = AsyncMock(return_value=[])
    server.indexer.get_stats = Mock(return_value={"documents_indexed": 0})

    server.semantic_cache.get = AsyncMock(return_value=None)
    server.semantic_cache.set = AsyncMock()
    server.semantic_cache.get_stats = Mock(return_value={"hits": 0, "misses": 0})
    server.semantic_cache.clear = AsyncMock()

    server.indexer.remove_source = AsyncMock(return_value=True)

    server.embedding_manager.get_embedding = AsyncMock(return_value=Mock())
    server.embedding_manager.get_cache_stats = Mock(return_value={"hits": 0})
    server.embedding_manager.clear_cache = AsyncMock()

    server.redis_store.hierarchical_search = AsyncMock(
        return_value=[{"id": "1", "content": "Test content", "score": 0.9}]
    )

    server.knowledge_graph.build_from_documents = AsyncMock()
    server.knowledge_graph.query_subgraph = AsyncMock(
        return_value=Mock(entities=[], relationships=[], central_entities=[], metadata={})
    )
    server.knowledge_graph.get_graph_stats = Mock(return_value={"entity_count": 0})

    server.file_watcher.watch = AsyncMock(return_value="source_123")
    server.file_watcher.unwatch = AsyncMock(return_value=True)
    server.file_watcher.get_stats = Mock(return_value={"watched_sources": 0})

    return server
