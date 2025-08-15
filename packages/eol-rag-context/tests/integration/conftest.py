"""Integration test configuration and fixtures.

This version doesn't mock Redis since we need real Redis for integration tests.

"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

# Mock only non-essential dependencies for integration tests
# Don't mock modules that we have installed and need for tests
mock_modules = [
    "fastmcp",
    "fastmcp.server",
    "tree_sitter",
    "tree_sitter_python",
    "tree_sitter_javascript",
    "pypdf",
    "docx",
    # 'networkx',  # Don't mock - needed for knowledge graph
    "watchdog",
    "watchdog.observers",
    "watchdog.events",
    "typer",
    "rich",
    "rich.console",
    "rich.table",
    "gitignore_parser",
]

# Mock these modules
for module in mock_modules:
    if module not in sys.modules:
        sys.modules[module] = MagicMock()


# Create a mock for sentence_transformers that returns real numpy arrays
class MockSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.dimension = 384 if "MiniLM" in model_name else 768

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
        **kwargs,
    ):
        if isinstance(texts, str):
            texts = [texts]
        # Return consistent embeddings based on text hash for reproducibility
        embeddings = []
        for text in texts:
            # Generate deterministic embedding based on text
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.randn(self.dimension).astype(np.float32)
            # Normalize to unit length if requested
            if normalize_embeddings:
                embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        return np.array(embeddings) if len(embeddings) > 1 else embeddings[0]


# Create mock for openai
class MockOpenAIEmbeddings:
    def __init__(self):
        self.dimension = 1536

    def create(self, input, model):
        if isinstance(input, str):
            input = [input]
        embeddings = []
        for text in input:
            np.random.seed(hash(text) % 2**32)
            embedding = np.random.randn(self.dimension).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding.tolist())

        mock_response = Mock()
        mock_response.data = [Mock(embedding=emb) for emb in embeddings]
        return mock_response


mock_st = Mock()
mock_st.SentenceTransformer = MockSentenceTransformer
sys.modules["sentence_transformers"] = mock_st

mock_openai = Mock()
mock_openai.OpenAI = Mock
mock_openai_client = Mock()
mock_openai_client.embeddings = MockOpenAIEmbeddings()
mock_openai.OpenAI.return_value = mock_openai_client
sys.modules["openai"] = mock_openai

# Now import our modules - Redis will use real implementation
from eol.rag_context import (
    config,
    document_processor,
    embeddings,
    indexer,
    knowledge_graph,
    redis_client,
    semantic_cache,
)


@pytest.fixture(scope="function")
def event_loop():
    """Create an event loop for each test function."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def redis_config():
    """Redis configuration for integration tests."""
    return config.RedisConfig(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        password=None,
        max_connections=10,
    )


@pytest.fixture(scope="function")
async def redis_store(redis_config):
    """Create a Redis store for testing with real Redis.

    Note: This requires Redis Stack or Redis with RediSearch module loaded.
    Tests will fail if vector search is not available.

    """
    # Use default index config
    index_config = config.IndexConfig()

    store = redis_client.RedisVectorStore(redis_config, index_config)

    # Connect to real Redis
    await store.connect_async()
    store.connect()

    # Create indexes with correct dimension (384 for all-MiniLM-L6-v2)
    # This will fail if RediSearch module is not available - which is correct behavior
    try:
        store.create_hierarchical_indexes(embedding_dim=384)
    except Exception as e:
        # Re-raise if it's a module not found error
        if "unknown command" in str(e).lower() and "ft.create" in str(e).lower():
            pytest.skip(f"RediSearch module not available: {e}")
        raise e

    yield store

    # Cleanup - flush test data
    try:
        # Clean up test keys
        test_keys = await store.async_redis.keys("doc:*")
        if test_keys:
            await store.async_redis.delete(*test_keys)
        cache_keys = await store.async_redis.keys("cache:*")
        if cache_keys:
            await store.async_redis.delete(*cache_keys)
        await store.close()
    except Exception:
        pass


@pytest.fixture
async def embedding_manager():
    """Create an embedding manager for testing."""
    cfg = config.EmbeddingConfig(
        provider="sentence-transformers", model_name="all-MiniLM-L6-v2", dimension=384
    )
    return embeddings.EmbeddingManager(cfg)


@pytest.fixture
async def document_processor_instance():
    """Create a document processor for testing."""
    return document_processor.DocumentProcessor(
        config.DocumentConfig(), config.ChunkingConfig()
    )


@pytest.fixture
async def indexer_instance(redis_store, document_processor_instance, embedding_manager):
    """Create an indexer for testing."""
    return indexer.DocumentIndexer(
        config.RAGConfig(), document_processor_instance, embedding_manager, redis_store
    )


@pytest.fixture
async def semantic_cache_instance(redis_store, embedding_manager):
    """Create a semantic cache for testing."""
    cache_config = config.CacheConfig(
        enabled=True, similarity_threshold=0.9, max_cache_size=100, ttl_seconds=3600
    )
    return semantic_cache.SemanticCache(cache_config, embedding_manager, redis_store)


@pytest.fixture
async def knowledge_graph_instance(redis_store, embedding_manager):
    """Create a knowledge graph builder for testing."""
    # KnowledgeGraphBuilder needs both redis_store and embedding_manager
    return knowledge_graph.KnowledgeGraphBuilder(redis_store, embedding_manager)


@pytest.fixture
async def file_watcher_instance(indexer_instance):
    """Create a file watcher for testing."""
    from eol.rag_context.file_watcher import FileWatcher

    # Create file watcher with test-friendly settings
    watcher = FileWatcher(
        indexer=indexer_instance,
        graph_builder=None,  # No graph builder for simple tests
        debounce_seconds=0.5,  # Short debounce for testing
        batch_size=5,
        use_polling=True,  # Use polling to avoid watchdog dependencies
    )

    yield watcher

    # Cleanup
    await watcher.stop()


@pytest.fixture
async def temp_test_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)

        # Create test files
        (test_dir / "test.py").write_text(
            """
def hello_world():
    '''A simple hello world function.'''
    return "Hello, World!"

class TestClass:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value
"""
        )

        (test_dir / "README.md").write_text(
            """
# Test Project

This is a test project for integration testing.

## Features
- Feature 1
- Feature 2

## Installation
```bash
pip install test-project
```
"""
        )

        (test_dir / "config.json").write_text(
            """
{
    "name": "test-project",
    "version": "1.0.0",
    "settings": {
        "debug": true,
        "timeout": 30
    }
}
"""
        )

        (test_dir / "data.txt").write_text(
            """
This is a simple text file with some content.
It has multiple lines.
And various information.
"""
        )

        # Create subdirectory
        (test_dir / "src").mkdir()
        (test_dir / "src" / "module.py").write_text(
            """
import os

def process_data(data):
    return data.upper()
"""
        )

        yield test_dir
