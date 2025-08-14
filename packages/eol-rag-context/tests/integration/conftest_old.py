"""
Integration test configuration and fixtures.
"""

import asyncio
import os

# Only mock what's absolutely necessary for missing packages
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock all external dependencies that aren't installed
mock_modules = [
    "fastmcp",
    "fastmcp.server",
    "tree_sitter",
    "tree_sitter_python",
    "tree_sitter_javascript",
    "magic",
    "pypdf",
    "docx",
    "sentence_transformers",
    "openai",
    "networkx",
    "yaml",
    "bs4",
    "markdown",
    "watchdog",
    "watchdog.observers",
    "watchdog.events",
    "typer",
    "rich",
    "rich.console",
    "rich.table",
    "gitignore_parser",
]
for module in mock_modules:
    if module not in sys.modules:
        sys.modules[module] = MagicMock()

from eol.rag_context import (
    config,
    document_processor,
    embeddings,
    file_watcher,
    indexer,
    knowledge_graph,
    redis_client,
    semantic_cache,
    server,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def redis_config():
    """Redis configuration for integration tests."""
    return config.RedisConfig(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        password=None,
        max_connections=10,
    )


@pytest.fixture(scope="session")
async def redis_store(redis_config):
    """Create a Redis store for testing."""
    store = redis_client.RedisVectorStore(redis_config, config.IndexConfig())

    # Connect to Redis
    max_retries = 10
    for i in range(max_retries):
        try:
            await store.connect_async()
            store.connect()  # Also establish sync connection
            break
        except Exception as e:
            if i == max_retries - 1:
                raise
            print(f"Waiting for Redis... attempt {i+1}/{max_retries}")
            await asyncio.sleep(2)

    # Create indexes
    store.create_hierarchical_indexes(embedding_dim=768)

    yield store

    # Cleanup
    await store.close()


@pytest.fixture
async def embedding_manager():
    """Create an embedding manager for testing."""
    cfg = config.EmbeddingConfig(
        provider="sentence_transformers", model_name="all-MiniLM-L6-v2", dimension=384
    )
    return embeddings.EmbeddingManager(cfg)


@pytest.fixture
async def document_processor_instance():
    """Create a document processor for testing."""
    return document_processor.DocumentProcessor(config.DocumentConfig(), config.ChunkingConfig())


@pytest.fixture
async def indexer_instance(redis_store, document_processor_instance, embedding_manager):
    """Create an indexer for testing."""
    return indexer.DocumentIndexer(
        config.RAGConfig(), document_processor_instance, embedding_manager, redis_store
    )


@pytest.fixture
async def semantic_cache_instance(redis_store, embedding_manager):
    """Create a semantic cache for testing."""
    cache_config = config.SemanticCacheConfig(
        enabled=True,
        similarity_threshold=0.9,
        max_cache_size=100,
        ttl_seconds=3600,
        target_hit_rate=0.31,
    )
    return semantic_cache.SemanticCache(cache_config, embedding_manager, redis_store)


@pytest.fixture
async def knowledge_graph_instance(redis_store):
    """Create a knowledge graph builder for testing."""
    kg_config = config.KnowledgeGraphConfig(enabled=True, max_depth=3, similarity_threshold=0.8)
    return knowledge_graph.KnowledgeGraphBuilder(kg_config, redis_store)


@pytest.fixture
async def file_watcher_instance(indexer_instance):
    """Create a file watcher for testing."""
    watcher_config = config.FileWatcherConfig(enabled=True, watch_interval=1, debounce_seconds=0.5)
    return file_watcher.FileWatcher(watcher_config, indexer_instance)


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


@pytest.fixture
def mock_fastmcp():
    """Mock FastMCP for server tests."""
    from unittest.mock import AsyncMock, MagicMock

    mock_mcp = MagicMock()
    mock_mcp.tool = MagicMock()
    mock_mcp.run = AsyncMock()

    return mock_mcp
