# Test Failures Fix Plan

## Investigation Summary

Found 5 test failures across 2 test files:

### 1. test_parallel_indexer.py (4 failures)

- **2 Errors**: `TestParallelFileScanner` initialization issues
- **2 Failures**: `TestParallelIndexer` checkpoint method issues

### 2. test_server_simple.py (1 failure)

- **1 Failure**: Error handling test expecting different behavior

## Root Causes

### Issue 1: ParallelFileScanner Constructor Mismatch

**Location**: tests/unit/test_parallel_indexer.py:120
**Problem**: Test fixture creates `ParallelFileScanner(config)` but class requires `ParallelFileScanner(config, rag_config)`
**Impact**: 2 test errors

### Issue 2: ParallelIndexer Redis Attribute Access

**Location**: tests/unit/test_parallel_indexer.py:317, 334
**Problem**: Tests expect `parallel_indexer.redis.redis.hset` but ParallelIndexer inherits from DocumentIndexer which might have different attribute structure
**Impact**: 2 test failures

### Issue 3: Server index_directory Behavior Change

**Location**: tests/unit/test_server_simple.py:330
**Problem**: Test expects error status when indexer is None, but method now initializes components automatically (line 1254: `await self.initialize()`)
**Impact**: 1 test failure

## Fix Plan

### Fix 1: Update ParallelFileScanner Test Fixture

```python
# tests/unit/test_parallel_indexer.py:118-120
@pytest.fixture
def scanner(self, config):
    """Create file scanner."""
    rag_config = RAGConfig()  # Add missing RAGConfig
    return ParallelFileScanner(config, rag_config)
```

### Fix 2: Fix ParallelIndexer Checkpoint Tests

Need to check actual attribute structure and update mocks:

```python
# Option A: If redis is stored directly
parallel_indexer.redis.hset = AsyncMock()

# Option B: If redis has nested structure
parallel_indexer.redis_store.redis.hset = AsyncMock()
```

### Fix 3: Update Server Error Handling Test

The test needs to prevent auto-initialization or mock it to fail:

```python
# Option A: Mock initialize to raise exception
with patch.object(srv, 'initialize', side_effect=Exception("Init failed")):
    result = await srv.index_directory("/test/path")
    assert result["status"] == "error"

# Option B: Update test expectation to match new behavior
# If indexer is None, it auto-initializes and starts indexing
assert result["status"] == "started"  # Not "error"
```

## Implementation Order

1. **Fix ParallelFileScanner tests** (simplest fix)
   - Add RAGConfig to fixture
   - Import RAGConfig in test file

2. **Fix ParallelIndexer checkpoint tests**
   - Investigate actual redis attribute structure
   - Update mock paths accordingly

3. **Fix server error handling test**
   - Decide on behavior: should it auto-initialize or return error?
   - Update test to match intended behavior

## Testing Strategy

1. Fix each issue individually
2. Run specific test after each fix
3. Run full test suite after all fixes
4. Verify no regressions in other tests

## Estimated Impact

- These fixes are isolated to test code only
- No production code changes required
- Tests are failing due to:
  - Missing test dependencies (RAGConfig)
  - Incorrect mock attribute paths
  - Changed behavior in server (auto-initialization)
