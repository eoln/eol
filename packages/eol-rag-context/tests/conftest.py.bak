"""
Pytest configuration and fixtures.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, AsyncGenerator
import numpy as np

from eol.rag_context.config import RAGConfig, RedisConfig, EmbeddingConfig
from eol.rag_context.redis_client import RedisVectorStore
from eol.rag_context.embeddings import EmbeddingManager
from eol.rag_context.document_processor import DocumentProcessor
from eol.rag_context.indexer import DocumentIndexer


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


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
    config.embedding.dimension = 128
    config.embedding.model_name = "all-MiniLM-L6-v2"
    
    # Disable cache for tests
    config.cache.enabled = False
    
    return config


@pytest.fixture
def redis_config() -> RedisConfig:
    """Redis configuration for tests."""
    return RedisConfig(
        host="localhost",
        port=6379,
        db=15  # Use separate DB for tests
    )


@pytest.fixture
async def redis_store(redis_config: RedisConfig, test_config: RAGConfig) -> AsyncGenerator[RedisVectorStore, None]:
    """Create Redis store for tests."""
    store = RedisVectorStore(redis_config, test_config.index)
    await store.connect_async()
    
    # Clean test DB
    await store.async_redis.flushdb()
    
    yield store
    
    # Cleanup
    await store.async_redis.flushdb()
    await store.close()


@pytest.fixture
def mock_embedding_manager(test_config: RAGConfig) -> EmbeddingManager:
    """Create mock embedding manager."""
    class MockEmbeddingProvider:
        async def embed(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            # Return random embeddings for testing
            return np.random.randn(len(texts), test_config.embedding.dimension)
        
        async def embed_batch(self, texts, batch_size=32):
            return await self.embed(texts)
    
    manager = EmbeddingManager(test_config.embedding)
    manager.provider = MockEmbeddingProvider()
    return manager


@pytest.fixture
def sample_documents(temp_dir: Path) -> Dict[str, Path]:
    """Create sample documents for testing."""
    docs = {}
    
    # Markdown file
    md_file = temp_dir / "test.md"
    md_file.write_text("""# Test Document

## Introduction
This is a test document for RAG indexing.

## Code Example
```python
def hello_world():
    print("Hello, World!")
```

## Conclusion
This concludes our test document.
""")
    docs["markdown"] = md_file
    
    # Python file
    py_file = temp_dir / "test.py"
    py_file.write_text("""
def factorial(n):
    \"\"\"Calculate factorial of n.\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n - 1)

class Calculator:
    \"\"\"Simple calculator class.\"\"\"
    
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b
""")
    docs["python"] = py_file
    
    # JSON file
    json_file = temp_dir / "config.json"
    json_file.write_text("""{
    "name": "test-project",
    "version": "1.0.0",
    "dependencies": {
        "redis": "^5.0.0",
        "numpy": "^1.24.0"
    }
}""")
    docs["json"] = json_file
    
    # Text file
    txt_file = temp_dir / "readme.txt"
    txt_file.write_text("""This is a sample project for testing the RAG system.

It includes various file types to test the document processor.
The system should handle markdown, code, JSON, and plain text files.

Features:
- Document indexing
- Vector search
- Knowledge graph
- Real-time updates
""")
    docs["text"] = txt_file
    
    return docs


@pytest.fixture
async def indexed_documents(
    redis_store: RedisVectorStore,
    mock_embedding_manager: EmbeddingManager,
    test_config: RAGConfig,
    sample_documents: Dict[str, Path]
) -> DocumentIndexer:
    """Index sample documents for testing."""
    processor = DocumentProcessor(test_config.document, test_config.chunking)
    indexer = DocumentIndexer(
        test_config,
        processor,
        mock_embedding_manager,
        redis_store
    )
    
    # Index all sample documents
    for doc_path in sample_documents.values():
        await indexer.index_file(doc_path)
    
    return indexer


# Skip markers for tests requiring Redis
pytest.mark.redis = pytest.mark.skipif(
    not pytest.config.getoption("--redis"),
    reason="Redis tests not enabled (use --redis flag)"
)


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--redis",
        action="store_true",
        default=False,
        help="Run tests requiring Redis"
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )


from typing import Dict  # Add this import at the top