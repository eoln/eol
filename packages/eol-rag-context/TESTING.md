# Testing Guide

## Quick Start

```bash
# Run all tests with Redis
./test_all.sh

# Unit tests only (no Redis needed)
uv run pytest tests/test_*.py

# Integration tests (Redis required)
docker run -d -p 6379:6379 redis:8.2-alpine
uv run pytest tests/integration/
```

## Prerequisites

- **Python 3.13+** (required)
- **Redis 8.2+** with native Vector Sets (for integration tests)
- **Docker** (for running Redis)

## Test Structure

### Unit Tests (43% coverage)

Location: `tests/test_*.py`

- `test_config.py` - Configuration management
- `test_embeddings.py` - Embedding providers
- `test_document_processor.py` - Document processing
- `test_indexer.py` - Indexing operations

All external dependencies are mocked in unit tests.

### Integration Tests (+37% coverage)

Location: `tests/integration/test_*.py`

- `test_redis_integration.py` - Redis vector operations (10 tests)
- `test_document_processing_integration.py` - File processing (9 tests)
- `test_indexing_integration.py` - Indexing workflows (10 tests)
- `test_full_workflow_integration.py` - End-to-end RAG (7 tests)
- `test_tutorial_examples.py` - Example validation (16 tests)

**Total**: 52 integration tests, 100% pass rate ✅

## Coverage Requirements

- **Target**: 80% minimum
- **Current**: 88.5% ✅
- **CI/CD**: Enforced via GitHub Actions

## Running Tests

### With Docker (Recommended)

```bash
# Start Redis
docker compose -f docker-compose.test.yml up -d

# Run tests
uv run pytest

# Stop Redis
docker compose -f docker-compose.test.yml down
```

### Manual Redis Setup

```bash
# Start Redis 8.2+
docker run -d -p 6379:6379 redis:8.2-alpine

# Verify Redis is running
redis-cli ping  # Should return PONG

# Run tests
uv run pytest tests/
```

## Integration Testing Rules

### Never Mock Core Dependencies

- ❌ Don't mock Redis connections
- ❌ Don't mock vector operations
- ❌ Don't create fallback tests that hide failures
- ✅ Use real Redis with vector support
- ✅ Use real numpy arrays (384 dimensions)
- ✅ Let tests fail if dependencies are missing

### Key Conventions

**Redis Key Format**:

- Level 1: `concept:{doc_id}`
- Level 2: `section:{doc_id}`
- Level 3: `chunk:{doc_id}`

**Vector Search Returns**:

```python
# Returns: List[Tuple[str, float, Dict[str, Any]]]
results = await redis_store.vector_search(query_embedding, k=5)
for doc_id, score, data in results:
    # Process results
```

**Embedding Dimensions**:

- Use 384 for all-MiniLM-L6-v2
- Use 768 for larger models

## Common Issues

### Import Errors

```bash
# Activate virtual environment
source .venv/bin/activate
```

### Redis Connection Failed

```bash
# Check Redis is running
redis-cli ping

# Start Redis if needed
docker run -d -p 6379:6379 redis:8.2-alpine
```

### Python Version Error

```bash
# Must use Python 3.13+
python --version
```

## Performance Targets

- **Indexing**: >10 documents/second
- **Search**: >20 searches/second
- **Cache**: >50 operations/second
- **Response**: <500ms for typical queries

## CI/CD Pipeline

Tests run automatically on:

- Pull requests
- Pushes to main branch
- GitHub Actions with quality gates

Quality gates enforce:

- 80% minimum coverage
- All integration tests passing
- Performance benchmarks met
- Security scans clean
