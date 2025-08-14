# Test Failure Analysis and Fix Plan

## Current Status

- **Total Tests**: 52 integration tests
- **Passing**: 15 tests
- **Failing**: 35 tests
- **Skipped**: 2 tests

## Phase 1: Discovery and Analysis

### Step 1.1: Collect Failure Patterns

**Goal**: Identify all unique failure types and group tests by root cause

**Actions**:

1. Run each test file individually to capture detailed errors
2. Create failure taxonomy document
3. Map each test to its failure category

**Commands**:

```bash
source .venv/bin/activate
pytest tests/integration/test_document_processing_integration.py -v --tb=short > failures/doc_processing.log
pytest tests/integration/test_indexing_integration.py -v --tb=short > failures/indexing.log
pytest tests/integration/test_full_workflow_integration.py -v --tb=short > failures/workflow.log
pytest tests/integration/test_tutorial_examples.py -v --tb=short > failures/tutorial.log
```

### Step 1.2: Categorize Failures

**Expected Categories**:

1. **Missing Fields**: Tests expect fields that aren't provided
   - `metadata` field missing in chunks
   - Missing attributes on returned objects

2. **Type Mismatches**: Return type doesn't match expectation
   - Methods returning `int` instead of result objects
   - Methods returning `dict` instead of async results

3. **API Mismatches**: Method names or signatures changed
   - `index_directory` vs `index_folder`
   - Missing methods on server object

4. **Async/Await Issues**: Incorrect async patterns
   - Using `await` on synchronous methods
   - Not using `await` on async methods

5. **Redis Operation Failures**: Vector search or storage issues
   - TAG field filtering problems
   - Key format mismatches

## Phase 2: API Contract Documentation

### Step 2.1: Document Expected Contracts

**Goal**: Create clear specification of what each API should return

**Template for each failing method**:

```python
# Method: ClassName.method_name
# Test expects:
#   - Return type: TypeName
#   - Fields: field1, field2, field3
#   - Behavior: description
#
# Current implementation:
#   - Return type: ActualType
#   - Fields: actual_field1, actual_field2
#   - Issues: what's wrong
```

### Step 2.2: Decision Matrix

For each mismatch, decide:

- Fix the implementation to match tests (preferred)
- Fix the tests to match implementation (if implementation is correct)
- Create adapter/wrapper to bridge the gap

## Phase 3: Implementation Fixes

### Step 3.1: Fix Document Processor Metadata (Priority 1)

**Problem**: Chunks missing `metadata` field
**Files**: `src/eol/rag_context/document_processor.py`

**Fix Strategy**:

1. Locate all chunk creation points
2. Add metadata field with:
   - Source file path
   - Chunk index
   - Creation timestamp
   - Parent document ID
   - Processing parameters

**Implementation**:

```python
def _create_chunk(self, content, chunk_type, **kwargs):
    return {
        "content": content,
        "type": chunk_type,
        "metadata": {
            "chunk_index": kwargs.get("index", 0),
            "source": kwargs.get("source", ""),
            "timestamp": time.time(),
            "parent_id": kwargs.get("parent_id", ""),
        },
        "tokens": len(content.split()),
    }
```

### Step 3.2: Fix Indexer Return Types (Priority 2)

**Problem**: `index_file` returns `int` but tests expect result object
**Files**: `src/eol/rag_context/indexer.py`

**Fix Strategy**:

1. Create `IndexResult` dataclass
2. Modify methods to return result objects
3. Preserve backward compatibility where needed

**Implementation**:

```python
@dataclass
class IndexResult:
    source_id: str
    chunks: int
    files: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Step 3.3: Fix Tutorial Examples API (Priority 3)

**Problem**: Server missing expected methods
**Files**: `src/eol/rag_context/server.py`

**Fix Strategy**:

1. Map all missing methods
2. Implement missing methods or create aliases
3. Ensure MCP protocol compliance

### Step 3.4: Fix Async/Await Patterns (Priority 4)

**Problem**: Inconsistent async usage
**Files**: Multiple

**Fix Strategy**:

1. Audit all Redis operations
2. Check which redis-py methods are sync vs async
3. Use proper async client where needed
4. Remove unnecessary awaits

## Phase 4: Validation

### Step 4.1: Incremental Testing

**Goal**: Verify each fix without breaking others

**Process**:

1. Fix one category at a time
2. Run affected tests
3. Ensure no regression in passing tests
4. Document any API changes

**Commands**:

```bash
# After each fix
pytest tests/integration/test_redis_integration.py -v  # Should stay green
pytest tests/integration/[fixed_test_file] -v  # Should improve
```

### Step 4.2: Full Test Suite Run

**Goal**: Achieve >80% test pass rate

**Commands**:

```bash
# Final validation
pytest tests/integration/ -v --tb=short
pytest tests/ --cov=eol.rag_context --cov-report=term
```

## Phase 5: Documentation Update

### Step 5.1: Update API Documentation

- Document all return types
- Add examples for each method
- Update docstrings

### Step 5.2: Update Test Documentation

- Document test requirements
- Add troubleshooting guide
- Update CI/CD configuration

## Execution Timeline

### Day 1: Analysis (2-3 hours)

- [ ] Run all tests individually
- [ ] Create failure taxonomy
- [ ] Document API contracts

### Day 2: Core Fixes (3-4 hours)

- [ ] Fix document processor metadata
- [ ] Fix indexer return types
- [ ] Test core functionality

### Day 3: API Fixes (2-3 hours)

- [ ] Fix tutorial examples
- [ ] Fix async patterns
- [ ] Run integration tests

### Day 4: Validation (1-2 hours)

- [ ] Full test suite run
- [ ] Update documentation
- [ ] Create PR

## Success Criteria

1. **Minimum 80% tests passing** (42+ out of 52)
2. **All Redis integration tests remain green**
3. **No regression in currently passing tests**
4. **Clear documentation of any API changes**
5. **CI/CD pipeline passes**

## Risk Mitigation

1. **Create backup branch** before making changes
2. **Test each fix in isolation**
3. **Keep detailed changelog** of modifications
4. **Have rollback plan** if fixes break production code
5. **Coordinate with team** on API changes

## Rollback Plan

If fixes cause issues:

1. Revert to backup branch
2. Create feature flags for new behavior
3. Implement fixes behind flags
4. Gradually enable with testing

## Notes

- Prioritize fixes that unblock the most tests
- Some tests may need to be updated rather than fixing code
- Consider creating adapters for backward compatibility
- Document all decisions for future reference
