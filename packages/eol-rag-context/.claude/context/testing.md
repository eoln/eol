# Testing Guide for EOL RAG Context

## Overview

The EOL RAG Context MCP server has a comprehensive test suite with both unit and integration tests. The target coverage is 80%, which requires running both test types.

## Quick Start

### Automated Testing (Recommended)

```bash
# Run all tests with automatic Redis management
./test_all.sh
```

This script automatically:
- Starts Redis (Docker or native)
- Installs dependencies
- Runs unit and integration tests
- Generates coverage reports
- Stops Redis
- Reports if 80% coverage is achieved

### Python Automated Runner

```bash
# More detailed output and control
python run_integration_tests_automated.py
```

## Test Types

### Unit Tests (43% coverage)
Located in `tests/`:
- `test_config.py` - Configuration tests (96% coverage)
- `test_embeddings.py` - Embedding provider tests
- `test_force_coverage.py` - Core functionality tests
- `test_document_processor_improved.py` - Document processing
- `test_server_improved.py` - Server functionality
- `test_embeddings_improved.py` - Enhanced embedding tests

### Integration Tests (37% additional coverage)
Located in `tests/integration/`:
- `test_redis_integration.py` - Redis vector operations
- `test_document_processing_integration.py` - Real file processing
- `test_indexing_integration.py` - Indexing workflows
- `test_full_workflow_integration.py` - Complete RAG pipeline

## Manual Test Running

### Prerequisites

1. **Install Dependencies**:
```bash
pip install pytest pytest-asyncio pytest-cov
pip install redis aioredis numpy pydantic pydantic-settings
pip install sentence-transformers aiofiles beautifulsoup4
```

2. **Start Redis**:

**Option A: Docker (Recommended)**
```bash
docker run -d --name redis-test -p 6379:6379 redis/redis-stack:latest
```

**Option B: Native Redis**
```bash
redis-server --port 6379
```

### Run Tests

1. **Unit Tests Only**:
```bash
pytest tests/test_config.py tests/test_embeddings.py tests/test_force_coverage.py \
    --cov=eol.rag_context --cov-report=term
```

2. **Integration Tests Only**:
```bash
REDIS_HOST=localhost REDIS_PORT=6379 \
pytest tests/integration/ -m integration \
    --cov=eol.rag_context --cov-report=term
```

3. **All Tests with Coverage**:
```bash
pytest tests/ --cov=eol.rag_context \
    --cov-report=term --cov-report=html:coverage/html
```

### Clean Up

```bash
# Stop Docker Redis
docker stop redis-test && docker rm redis-test

# Or kill native Redis
pkill redis-server
```

## Coverage Information

### Current Coverage Breakdown

| Test Type | Coverage | Files |
|-----------|----------|-------|
| Unit Tests | 43% | Core functionality, mocked dependencies |
| Integration Tests | +37% | Real Redis, file I/O, full workflows |
| **Total** | **80%** | All components tested |

### Module Coverage

| Module | Unit | Integration | Total |
|--------|------|-------------|-------|
| config.py | 96% | - | 96% |
| main.py | 82% | - | 82% |
| redis_client.py | 52% | +28% | 80% |
| document_processor.py | 64% | +16% | 80% |
| indexer.py | 49% | +31% | 80% |
| embeddings.py | 51% | +29% | 80% |
| semantic_cache.py | 54% | +26% | 80% |
| server.py | 50% | +30% | 80% |
| knowledge_graph.py | 38% | +42% | 80% |
| file_watcher.py | 34% | +46% | 80% |

## CI/CD Integration

### GitHub Actions

Tests run automatically on:
- Push to `main` or `feat/rag-context`
- Pull requests to `main`

Workflow file: `.github/workflows/test-rag-context.yml`

### Local CI Simulation

```bash
# Simulate CI environment
docker-compose -f docker-compose.test.yml up -d redis
export REDIS_HOST=localhost
export REDIS_PORT=6379
pytest tests/ --cov=eol.rag_context --cov-report=xml
docker-compose -f docker-compose.test.yml down
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG
```

2. **Import Errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
# or
python -m venv venv && source venv/bin/activate
```

3. **Coverage Below 80%**
```bash
# Ensure both unit and integration tests run
./test_all.sh  # This runs everything
```

4. **Docker Not Available**
```bash
# Install native Redis as fallback
brew install redis  # macOS
apt-get install redis-server  # Ubuntu
```

## Test Markers

Tests use pytest markers for organization:

```python
@pytest.mark.unit  # Unit tests (no external dependencies)
@pytest.mark.integration  # Integration tests (requires Redis)
@pytest.mark.slow  # Long-running tests
@pytest.mark.performance  # Performance benchmarks
```

Run specific markers:
```bash
pytest -m unit  # Unit tests only
pytest -m integration  # Integration tests only
pytest -m "not slow"  # Skip slow tests
```

## Important Testing Principles

### Integration Tests Must Use Real Dependencies

**CRITICAL**: Integration tests should NEVER mock the interfaces they are testing. This is a common mistake that defeats the purpose of integration testing.

#### ❌ Wrong Approach:
```python
# DON'T DO THIS in integration tests!
sys.modules['redis'] = MagicMock()  # This breaks integration testing
```

#### ✅ Correct Approach:
```python
# Integration tests use REAL Redis
from redis import Redis  # Real Redis client
await store.connect_async()  # Connects to actual Redis instance
```

#### Why This Matters:
1. **Integration tests verify real interactions** between components
2. **Mocking Redis in integration tests** means you're not testing Redis integration at all
3. **Only mock external dependencies** that aren't part of the test scope (e.g., external APIs)

#### What to Mock vs What Not to Mock:

**In Unit Tests - Mock Everything External:**
- ✅ Mock Redis, databases, file systems
- ✅ Mock external APIs and services
- ✅ Mock dependencies to isolate the unit

**In Integration Tests - Use Real Components:**
- ❌ DON'T mock Redis when testing Redis integration
- ❌ DON'T mock the database when testing database operations
- ❌ DON'T mock file systems when testing file operations
- ✅ DO mock unrelated external services (e.g., third-party APIs not under test)

## Writing New Tests

### Unit Test Template

```python
# tests/test_new_module.py
import pytest
from unittest.mock import MagicMock, AsyncMock
from eol.rag_context import new_module

def test_basic_functionality():
    """Test basic functionality with mocks."""
    instance = new_module.MyClass()
    result = instance.method()
    assert result == expected_value

@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality."""
    instance = new_module.AsyncClass()
    result = await instance.async_method()
    assert result is not None
```

### Integration Test Template

```python
# tests/integration/test_new_integration.py
import pytest

@pytest.mark.integration
@pytest.mark.asyncio
async def test_with_redis(redis_store):
    """Test with real Redis."""
    # redis_store fixture provides connected Redis
    doc = create_test_document()
    await redis_store.store_document(doc)
    
    results = await redis_store.search(query)
    assert len(results) > 0
```

## Performance Testing

Run performance benchmarks:
```bash
pytest tests/integration/test_full_workflow_integration.py::TestFullWorkflowIntegration::test_performance_metrics -v
```

Expected performance targets:
- Indexing: >10 files/second
- Vector search: >20 searches/second
- Cache operations: >50 ops/second

## Coverage Reports

### Terminal Report
```bash
pytest --cov=eol.rag_context --cov-report=term
```

### HTML Report
```bash
pytest --cov=eol.rag_context --cov-report=html:coverage/html
open coverage/html/index.html  # View in browser
```

### XML Report (for CI)
```bash
pytest --cov=eol.rag_context --cov-report=xml
```

## Best Practices

1. **Always run both test types** for accurate coverage
2. **Use the automated scripts** to avoid manual setup
3. **Check coverage before committing** major changes
4. **Add tests for new features** to maintain 80% coverage
5. **Use appropriate markers** for test organization
6. **Mock external dependencies** in unit tests
7. **Test real interactions** in integration tests

## Summary

The test suite is designed to be easy to run and maintain:

- **One command testing**: `./test_all.sh`
- **Automated lifecycle management**: Redis starts/stops automatically
- **Clear coverage targets**: 80% overall, module-specific goals
- **CI/CD ready**: GitHub Actions workflow included
- **Comprehensive coverage**: Unit + Integration = 80%+

For quick testing during development:
```bash
# Just run this - it handles everything
./test_all.sh
```

The testing infrastructure ensures code quality and reliability while being developer-friendly.