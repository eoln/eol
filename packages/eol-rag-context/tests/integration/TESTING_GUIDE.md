# Integration Testing Guide

## Overview

Integration tests verify that all components of the EOL RAG Context system work together correctly with real dependencies. These tests **do not use mocks** for critical infrastructure like Redis, ensuring that the system behaves correctly in production-like conditions.

## Prerequisites

### Required Software

1. **Python 3.11+**
   ```bash
   python3 --version  # Should be 3.11 or higher
   ```

2. **Redis Stack Server** (includes RediSearch module)
   ```bash
   # Install via Homebrew (macOS)
   brew tap redis-stack/redis-stack
   brew install --cask redis-stack-server
   
   # Or via Docker
   docker run -d -p 6379:6379 redis/redis-stack:latest
   ```

3. **libmagic** (for file type detection)
   ```bash
   brew install libmagic
   ```

### Python Dependencies

All required Python packages will be installed automatically by the setup script.

## Quick Start

### 1. Initial Setup (One-Time)

Run the setup script to install all dependencies and prepare the test environment:

```bash
./setup_test_environment.sh
```

This script will:
- Install system dependencies (libmagic, Redis Stack)
- Create and activate a Python virtual environment
- Install all Python dependencies
- Start Redis Stack Server
- Create test data files
- Verify all components are working

### 2. Running Integration Tests

After setup, run integration tests with:

```bash
./run_integration_tests.sh
```

This script will:
- Verify Redis Stack is running with RediSearch module
- Clear Redis data for clean test state
- Run all integration tests
- Generate coverage reports
- Provide detailed test results

## Test Structure

### Test Files

- `test_redis_integration.py` - Tests Redis vector store operations
- `test_document_processing_integration.py` - Tests document processing with real file types
- `test_indexing_integration.py` - Tests document indexing workflow
- `test_full_workflow_integration.py` - Tests complete RAG pipeline
- `test_tutorial_examples.py` - Validates all tutorial code examples

### Test Data

Test data is created in `tests/test_data/` and includes:
- Markdown documents (`.md`)
- Python source files (`.py`)
- JSON configuration files (`.json`)
- Plain text files (`.txt`)

## Understanding Test Results

### Successful Run

```
========================================
Test Results Summary
========================================
Total Tests: 52
✓ Passed: 52
========================================
All integration tests passed!
========================================
```

### Failed Tests

```
========================================
Test Results Summary
========================================
Total Tests: 52
✓ Passed: 48
✗ Failed: 4
========================================
Integration tests failed!
========================================

To debug failures:
1. Check test_results/integration.log for details
2. Run specific test: pytest tests/integration/test_name.py -xvs
3. Check Redis: redis-cli ping
4. Check modules: redis-cli MODULE LIST
```

## Test Coverage

The integration tests aim for 80% code coverage. Coverage reports are generated in multiple formats:

- **Terminal**: Displayed after test run
- **HTML**: `coverage/integration/index.html`
- **XML**: `test_results/coverage.xml`

## Debugging Test Failures

### 1. Check Redis Status

```bash
# Check if Redis is running
redis-cli ping

# Check loaded modules (should include "search")
redis-cli MODULE LIST

# Clear Redis data if needed
redis-cli FLUSHDB
```

### 2. Run Specific Test

```bash
# Activate virtual environment
source .venv/bin/activate

# Set Python path
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# Run single test file
pytest tests/integration/test_redis_integration.py -xvs

# Run single test method
pytest tests/integration/test_redis_integration.py::TestRedisIntegration::test_connection -xvs
```

### 3. Check Test Logs

Detailed logs are saved to `test_results/integration.log` after each run.

### 4. Common Issues

#### Redis Stack Not Running
```bash
# Stop any existing Redis
redis-cli shutdown

# Start Redis Stack
redis-stack-server --daemonize yes
```

#### RediSearch Module Not Available
- Ensure Redis Stack is installed (not regular Redis)
- Verify with: `redis-cli MODULE LIST | grep search`

#### File Type Detection Errors
- Install libmagic: `brew install libmagic`
- Reinstall python-magic: `pip install --force-reinstall python-magic`

#### Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -e .`

## Writing New Integration Tests

### Guidelines

1. **No Mocking of Core Dependencies**
   - Use real Redis connections
   - Use real file operations
   - Use real embeddings (can be deterministic for reproducibility)

2. **Proper Setup and Teardown**
   - Clean Redis data before/after tests
   - Remove temporary files
   - Reset any global state

3. **Test Real Workflows**
   ```python
   @pytest.mark.asyncio
   async def test_complete_workflow(redis_store, indexer_instance):
       # Index real documents
       result = await indexer_instance.index_folder(Path("test_data"))
       assert result.file_count > 0
       
       # Search with real embeddings
       results = await redis_store.vector_search(query_embedding, k=5)
       assert len(results) > 0
       
       # Verify actual data storage
       stored = await redis_store.async_redis.hgetall("doc:test")
       assert stored is not None
   ```

4. **Handle Missing Dependencies Gracefully**
   ```python
   @pytest.fixture
   async def redis_store(redis_config):
       store = RedisVectorStore(redis_config, IndexConfig())
       try:
           await store.connect_async()
           store.create_hierarchical_indexes(embedding_dim=384)
       except Exception as e:
           if "unknown command" in str(e) and "FT.CREATE" in str(e):
               pytest.skip(f"RediSearch module not available: {e}")
           raise e
       yield store
       await store.close()
   ```

## CI/CD Integration

For GitHub Actions or other CI systems:

```yaml
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
  
  - name: Set up Python
    uses: actions/setup-python@v4
    with:
      python-version: '3.11'
  
  - name: Install dependencies
    run: |
      pip install -e .
      pip install pytest pytest-asyncio pytest-cov
  
  - name: Run integration tests
    env:
      REDIS_HOST: localhost
      REDIS_PORT: 6379
    run: |
      pytest tests/integration/ --cov=eol.rag_context --cov-report=xml
```

## Best Practices

1. **Always Use Real Dependencies**
   - Integration tests should catch issues that unit tests with mocks would miss
   - Use actual Redis, actual file I/O, actual parsing libraries

2. **Test Data Should Be Realistic**
   - Use representative sample files
   - Include edge cases (empty files, large files, special characters)
   - Test various file formats

3. **Clean State Between Tests**
   - Each test should start with a clean Redis database
   - Remove any temporary files created during tests
   - Reset any modified configuration

4. **Performance Considerations**
   - Integration tests are slower than unit tests (this is expected)
   - Use parallel test execution where possible
   - Cache dependencies between runs

5. **Documentation**
   - Document any special setup requirements
   - Explain what each test verifies
   - Provide troubleshooting steps for common failures

## Support

If you encounter issues:

1. Check this guide for troubleshooting steps
2. Review test logs in `test_results/integration.log`
3. Check Redis Stack documentation: https://redis.io/docs/stack/
4. File an issue with detailed error messages and environment information