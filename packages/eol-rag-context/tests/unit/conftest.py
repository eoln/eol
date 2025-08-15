"""
Pytest configuration and fixtures for unit tests.
"""

import importlib.machinery
import shutil
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import numpy as np
import pytest

from eol.rag_context.config import RAGConfig, RedisConfig
from eol.rag_context.document_processor import DocumentProcessor
from eol.rag_context.indexer import DocumentIndexer


class MockMultiDiGraph:
    """Mock implementation of NetworkX MultiDiGraph for testing."""

    def __init__(self):
        self._nodes = {}
        self._edges = []
        self.nodes = self._nodes

    def __len__(self):
        """Return the number of nodes (NetworkX convention)."""
        return len(self._nodes)

    def __contains__(self, node_id):
        """Check if node exists in graph."""
        return node_id in self._nodes

    def __iter__(self):
        """Iterate over nodes."""
        return iter(self._nodes)

    def add_node(self, node_id, **attrs):
        self._nodes[node_id] = attrs

    def add_edge(self, source, target, **attrs):
        self._edges.append((source, target, attrs))
        # Ensure both nodes exist
        if source not in self._nodes:
            self._nodes[source] = {}
        if target not in self._nodes:
            self._nodes[target] = {}

    def has_node(self, node_id):
        return node_id in self._nodes

    def has_edge(self, source, target):
        return any(e[0] == source and e[1] == target for e in self._edges)

    def remove_node(self, node_id):
        if node_id in self._nodes:
            del self._nodes[node_id]
            self._edges = [(s, t, a) for s, t, a in self._edges if s != node_id and t != node_id]

    def number_of_nodes(self):
        return len(self._nodes)

    def number_of_edges(self):
        return len(self._edges)

    def neighbors(self, node_id):
        return [e[1] for e in self._edges if e[0] == node_id]

    def degree(self, node_id=None):
        if node_id is None:
            # Return iterator of (node, degree) pairs for all nodes
            degrees = []
            for n in self._nodes:
                d = sum(1 for e in self._edges if e[0] == n or e[1] == n)
                degrees.append((n, d))
            return iter(degrees)
        else:
            # Return degree for specific node
            return sum(1 for e in self._edges if e[0] == node_id or e[1] == node_id)

    def edges(self, data=False):
        """Return edges, optionally with data."""
        if data:
            return [(e[0], e[1], e[2]) for e in self._edges]
        else:
            return [(e[0], e[1]) for e in self._edges]

    def in_degree(self, node_id=None):
        """Return in-degree of node(s)."""
        if node_id is None:
            degrees = []
            for n in self._nodes:
                d = sum(1 for e in self._edges if e[1] == n)
                degrees.append((n, d))
            return iter(degrees)
        else:
            return sum(1 for e in self._edges if e[1] == node_id)

    def out_degree(self, node_id=None):
        """Return out-degree of node(s)."""
        if node_id is None:
            degrees = []
            for n in self._nodes:
                d = sum(1 for e in self._edges if e[0] == n)
                degrees.append((n, d))
            return iter(degrees)
        else:
            return sum(1 for e in self._edges if e[0] == node_id)


@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """
    Automatically mock external dependencies for all unit tests.
    This fixture runs for every test with proper setup/teardown.
    """
    # Save original modules
    original_modules = {}
    modules_to_mock = [
        "networkx",
        "redis",
        "redis.asyncio",
        "redis.commands",
        "redis.commands.search",
        "redis.commands.search.field",
        "redis.commands.search.indexDefinition",
        "redis.commands.search.query",
        "redis.exceptions",
        "watchdog",
        "watchdog.observers",
        "watchdog.events",
        "fastmcp",
    ]

    for module_name in modules_to_mock:
        if module_name in sys.modules:
            original_modules[module_name] = sys.modules[module_name]

    # Create NetworkX mock
    nx_mock = MagicMock()
    nx_mock.__spec__ = importlib.machinery.ModuleSpec("networkx", None)
    nx_mock.MultiDiGraph = MockMultiDiGraph
    nx_mock.shortest_path = MagicMock(return_value=["node1", "node2", "node3"])
    sys.modules["networkx"] = nx_mock

    # Create Redis mocks
    mock_redis = MagicMock()
    mock_redis.__spec__ = importlib.machinery.ModuleSpec("redis", None)
    mock_redis.asyncio = MagicMock()
    mock_redis.asyncio.__spec__ = importlib.machinery.ModuleSpec("redis.asyncio", None)
    mock_redis.commands = MagicMock()
    mock_redis.commands.__spec__ = importlib.machinery.ModuleSpec("redis.commands", None)
    mock_redis.commands.search = MagicMock()
    mock_redis.commands.search.__spec__ = importlib.machinery.ModuleSpec(
        "redis.commands.search", None
    )
    mock_redis.commands.search.field = MagicMock()
    mock_redis.commands.search.field.__spec__ = importlib.machinery.ModuleSpec(
        "redis.commands.search.field", None
    )
    mock_redis.commands.search.indexDefinition = MagicMock()
    mock_redis.commands.search.indexDefinition.__spec__ = importlib.machinery.ModuleSpec(
        "redis.commands.search.indexDefinition", None
    )
    mock_redis.commands.search.query = MagicMock()
    mock_redis.commands.search.query.__spec__ = importlib.machinery.ModuleSpec(
        "redis.commands.search.query", None
    )
    mock_redis.exceptions = MagicMock()
    mock_redis.exceptions.__spec__ = importlib.machinery.ModuleSpec("redis.exceptions", None)

    sys.modules["redis"] = mock_redis
    sys.modules["redis.asyncio"] = mock_redis.asyncio
    sys.modules["redis.commands"] = mock_redis.commands
    sys.modules["redis.commands.search"] = mock_redis.commands.search
    sys.modules["redis.commands.search.field"] = mock_redis.commands.search.field
    sys.modules["redis.commands.search.indexDefinition"] = (
        mock_redis.commands.search.indexDefinition
    )
    sys.modules["redis.commands.search.query"] = mock_redis.commands.search.query
    sys.modules["redis.exceptions"] = mock_redis.exceptions

    # Create watchdog mocks
    watchdog_mock = MagicMock()
    watchdog_mock.__spec__ = importlib.machinery.ModuleSpec("watchdog", None)
    watchdog_observers_mock = MagicMock()
    watchdog_observers_mock.__spec__ = importlib.machinery.ModuleSpec("watchdog.observers", None)
    watchdog_events_mock = MagicMock()
    watchdog_events_mock.__spec__ = importlib.machinery.ModuleSpec("watchdog.events", None)

    # Create FileSystemEventHandler mock that can be instantiated
    class MockEventHandler:
        def __init__(self, *args, **kwargs):
            pass

    watchdog_events_mock.FileSystemEventHandler = MockEventHandler

    sys.modules["watchdog"] = watchdog_mock
    sys.modules["watchdog.observers"] = watchdog_observers_mock
    sys.modules["watchdog.events"] = watchdog_events_mock

    # Create fastmcp mock
    fastmcp_mock = MagicMock()
    fastmcp_mock.__spec__ = importlib.machinery.ModuleSpec("fastmcp", None)
    fastmcp_mock.FastMCP = MagicMock()
    fastmcp_mock.Context = MagicMock()
    sys.modules["fastmcp"] = fastmcp_mock

    # Yield control to test
    yield

    # Cleanup: restore original modules
    for module_name in modules_to_mock:
        if module_name in original_modules:
            sys.modules[module_name] = original_modules[module_name]
        elif module_name in sys.modules:
            del sys.modules[module_name]


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
    return RedisConfig(host="localhost", port=6379, db=15)  # Use separate DB for tests


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
