# Redis Setup Guide for Running Tests

**Date**: August 13, 2025  
**Current Coverage with Redis**: 68.18%  
**Coverage without Redis**: 58.98%  
**Improvement**: +9.2% when Redis is running

## Summary

The test coverage significantly improves from 58.98% to 68.18% when Redis Stack Server is running. Many integration tests and Redis-dependent unit tests require a live Redis instance with the RediSearch module.

## Redis Setup Options

### Option 1: Redis Stack Server (Native - Recommended)

**Installation:**
```bash
# macOS
brew tap redis-stack/redis-stack
brew install --cask redis-stack-server

# Verify installation
which redis-stack-server
```

**Start Redis:**
```bash
# Start in background
redis-stack-server --daemonize yes

# Verify it's running
redis-cli ping  # Should return "PONG"

# Check RediSearch module is loaded
redis-cli MODULE LIST | grep search
```

**Stop Redis:**
```bash
redis-cli shutdown
```

### Option 2: Docker Compose (Automated)

**Using the provided docker-compose.test.yml:**
```bash
# Start Redis and run tests
docker-compose -f docker-compose.test.yml up

# Or just start Redis
docker-compose -f docker-compose.test.yml up redis -d

# Stop Redis
docker-compose -f docker-compose.test.yml down
```

### Option 3: Docker (Manual)

**Start Redis Stack with Docker:**
```bash
# Remove any existing container
docker rm -f eol-test-redis 2>/dev/null

# Start Redis Stack container
docker run -d \
  --name eol-test-redis \
  -p 6379:6379 \
  -p 8001:8001 \
  redis/redis-stack:latest

# Verify
docker exec eol-test-redis redis-cli ping
```

**Stop Redis:**
```bash
docker stop eol-test-redis
docker rm eol-test-redis
```

## Automated Test Scripts

### 1. test_all.sh (Comprehensive)
```bash
# Automatically starts Redis Stack Server and runs all tests
./test_all.sh
```
- Checks for Redis Stack Server
- Starts Redis if not running
- Clears Redis data for clean state
- Runs unit and integration tests
- Generates coverage reports

### 2. run_tests_with_redis.sh (Flexible)
```bash
# Tries Docker first, then native Redis
./run_tests_with_redis.sh
```
- Attempts Docker Redis first
- Falls back to native Redis
- Automatically handles cleanup

### 3. run_integration_tests.sh (Integration Only)
```bash
# Runs only integration tests with Redis
./run_integration_tests.sh
```

## Manual Test Execution

### With Virtual Environment
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Start Redis Stack Server
redis-stack-server --daemonize yes

# 3. Run all tests with coverage
python -m pytest tests/ \
  --cov=eol.rag_context \
  --cov-report=term-missing \
  --cov-report=html

# 4. Stop Redis when done
redis-cli shutdown
```

### Specific Test Categories
```bash
# Unit tests only (less Redis dependency)
python -m pytest tests/ -k "not integration" --cov=eol.rag_context

# Integration tests only (requires Redis)
python -m pytest tests/integration/ --cov=eol.rag_context

# Redis-specific tests
python -m pytest tests/integration/test_redis_integration.py -v
```

## Coverage Impact by Redis Availability

### Modules Requiring Redis

| Module | Coverage w/o Redis | Coverage w/ Redis | Improvement |
|--------|-------------------|-------------------|-------------|
| `redis_client.py` | 13.69% | 46.39% | +32.70% |
| `semantic_cache.py` | 61.09% | 71.04% | +9.95% |
| Integration Tests | 0% (errors) | ~60% | +60% |

### Test Failures Without Redis

- **52 test errors** due to Redis connection failures
- All integration tests fail
- Several unit tests that use Redis fail
- File watcher tests have unrelated mock issues

## Troubleshooting

### Issue: "Connection refused" errors
**Solution:** Redis is not running. Start Redis Stack Server:
```bash
redis-stack-server --daemonize yes
```

### Issue: "FT.CREATE command not found"
**Solution:** Regular Redis is running instead of Redis Stack. You need Redis with RediSearch module:
```bash
# Stop regular Redis
redis-cli shutdown

# Start Redis Stack
redis-stack-server --daemonize yes
```

### Issue: Docker daemon not running
**Solution:** Start Docker Desktop or use native Redis Stack Server instead.

### Issue: Port 6379 already in use
**Solution:** Check what's using the port:
```bash
lsof -i :6379
# Kill the process or stop Redis properly
redis-cli shutdown
```

## Best Practices

1. **Always use Redis Stack Server** (not regular Redis) for full feature support
2. **Clear Redis data before test runs** for consistent results:
   ```bash
   redis-cli FLUSHDB
   ```
3. **Use virtual environment** to avoid dependency conflicts
4. **Run automated test scripts** for proper setup/teardown
5. **Check Redis health** before running tests:
   ```bash
   redis-cli ping
   redis-cli MODULE LIST | grep search
   ```

## Quick Start Commands

```bash
# One-liner to start Redis and run tests
redis-stack-server --daemonize yes && source .venv/bin/activate && python -m pytest tests/ --cov=eol.rag_context -q

# Full test suite with cleanup
./test_all.sh

# Docker-based testing
docker-compose -f docker-compose.test.yml up
```

## Next Steps to Reach 80% Coverage

With Redis running (68.18% coverage), we need +11.82% more coverage:

1. Fix mock configuration issues in file_watcher tests
2. Add more unit tests for:
   - Document processor chunking strategies
   - Embeddings providers
   - MCP server tools
3. Fix failing integration tests
4. Add tests for uncovered Redis client methods

---

*For full coverage analysis, see: 20250813_test_coverage_analysis.md*