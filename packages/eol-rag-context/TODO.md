# Test Fix TODO List

## 🎯 Goal
Fix 35 failing integration tests to achieve >80% pass rate (42+ out of 52 tests)

## 📊 Current Status
- [ ] Total Tests: 52
- [ ] Passing: 15 (29%)
- [ ] Failing: 35 (67%)
- [ ] Target: 42+ (80%)

---

## Phase 1: Analysis & Setup ✅ COMPLETED

### Preparation
- [x] Start Docker Desktop
- [x] Start Redis Stack container (`docker compose -f docker-compose.test.yml up -d redis`)
- [x] Activate virtual environment (`source .venv/bin/activate`)
- [x] Verify Redis connection (`redis-cli ping`)

### Analysis
- [x] Run all test files individually
- [x] Collect failure patterns
- [x] Create failure taxonomy
- [x] Document root causes
- [x] Prioritize fixes by impact

---

## Phase 2: Priority 1 Fixes - Missing Fields (7 tests) ✅ COMPLETED

### Fix Document Processor Metadata
**File**: `src/eol/rag_context/document_processor.py`
**Impact**: Fixed 7 tests

- [x] Locate all chunk creation methods:
  - [x] `_chunk_text()` - line 504
  - [x] `_chunk_markdown()` 
  - [x] `_chunk_code()`
  - [x] Other chunking methods

- [x] Add metadata field to chunks:
  ```python
  "metadata": {
      "source": file_path,
      "chunk_index": index,
      "timestamp": time.time(),
      "language": detected_language,  # for code
      "section": section_name,  # for markdown
  }
  ```

- [x] Update chunk creation in:
  - [x] Line 520-524 (text chunks)
  - [x] Line 531-533 (continuation)
  - [x] Line 545-550 (overlap chunks)
  - [x] Markdown section chunks
  - [x] Code AST chunks

### Testing Priority 1
- [x] Run: `pytest tests/integration/test_document_processing_integration.py -v`
- [x] Results - 7 of 9 passed:
  - [x] test_process_text_file
  - [x] test_process_markdown_file
  - [x] test_process_python_file
  - [x] test_process_json_file (fixed doc_type)
  - [x] test_process_javascript_file
  - [ ] test_chunking_strategies (partial fix)
  - [ ] test_concurrent_processing

---

## Phase 3: Priority 2 Fixes - Return Type Mismatches (10 tests) ✅ COMPLETED

### Create IndexResult Class
**File**: `src/eol/rag_context/indexer.py`
**Impact**: Fixed 6+ tests

- [x] Add IndexResult dataclass (after line 35):
  ```python
  @dataclass
  class IndexResult:
      source_id: str
      chunks: int = 0
      files: int = 0
      errors: List[str] = field(default_factory=list)
      metadata: Dict[str, Any] = field(default_factory=dict)
  ```

- [x] Update `index_file` method (line 365-371):
  - [x] Change return type annotation to `IndexResult`
  - [x] Return IndexResult object instead of int
  - [x] Include source_id, chunks, errors

- [x] Update `index_folder` method:
  - [x] Keep return type as IndexedSource (tests expect this)
  - [x] Add source_id parameter support
  - [x] Aggregate results from multiple files

- [x] Update related methods:
  - [x] Added indexed_files to IndexedSource
  - [x] Fixed async/await issues (partial Phase 4 work)

### Testing Priority 2
- [x] Run: `pytest tests/integration/test_indexing_integration.py -v`
- [x] Results - 6 of 10 passed:
  - [x] test_index_single_file
  - [x] test_index_folder
  - [x] test_concurrent_indexing
  - [ ] test_indexing_stats (still failing)
  - [x] test_metadata_extraction
  - [x] test_folder_scanner
  - [x] test_error_recovery

---

## Phase 4: Priority 3 Fixes - Async/Await Issues (6 tests)

### Fix Redis Async Operations
**File**: `src/eol/rag_context/indexer.py`
**Impact**: Fixes 6 tests

- [ ] Fix line 701 (already done):
  - [x] Remove `await` from `hset()`

- [ ] Fix line 709 (already done):
  - [x] Remove `await` from `expire()`

- [ ] Fix line 714:
  - [ ] Change: `data = await self.redis.redis.hgetall(source_key)`
  - [ ] To: `data = self.redis.redis.hgetall(source_key)`

- [ ] Audit other Redis operations:
  - [ ] Search for `await self.redis.redis`
  - [ ] Check redis-py docs for sync vs async methods
  - [ ] Remove unnecessary awaits

### Testing Priority 3
- [ ] Run: `pytest tests/integration/test_indexing_integration.py::TestIndexingIntegration::test_incremental_indexing -v`
- [ ] Expected to pass:
  - [ ] test_incremental_indexing
  - [ ] test_error_recovery
  - [ ] test_hierarchical_indexing

---

## Phase 5: Priority 4 Fixes - Type Conversion (3 tests)

### Handle Path/String Conversion
**File**: `src/eol/rag_context/indexer.py`
**Impact**: Fixes 3 tests

- [ ] Fix line 378 - Add type checking:
  ```python
  if isinstance(file_path, str):
      file_path = Path(file_path)
  file_path = file_path.resolve()
  ```

- [ ] Check other Path usages:
  - [ ] `index_folder()` method
  - [ ] `_scan_directory()` method
  - [ ] File metadata storage

- [ ] Fix NoneType Redis storage:
  - [ ] Add validation before storing in Redis
  - [ ] Convert None to empty string or skip

### Testing Priority 4
- [ ] Run: `pytest tests/integration/test_tutorial_examples.py::TestTutorialExamples::test_indexing_single_file -v`
- [ ] Expected to pass:
  - [ ] test_indexing_single_file
  - [ ] test_folder_scanner
  - [ ] test_indexing_directory

---

## Phase 6: Priority 5 Fixes - API Mismatches (5 tests)

### Fix Method Signatures
**Files**: Multiple
**Impact**: Fixes 5 tests

- [ ] Fix `index_folder` signature:
  - [ ] Add `source_id` parameter
  - [ ] Update docstring

- [ ] Fix doc_type values:
  - [ ] Map "json" → "structured"
  - [ ] Update type detection logic

- [ ] Fix missing server methods:
  - [ ] Add `index_directory` alias to `index_folder`
  - [ ] Add `watch_directory` method
  - [ ] Check other missing methods in tutorial tests

### Testing Priority 5
- [ ] Run: `pytest tests/integration/test_tutorial_examples.py -v`
- [ ] Expected improvements in tutorial example tests

---

## Phase 7: Final Validation

### Full Test Suite
- [ ] Run all integration tests:
  ```bash
  pytest tests/integration/ -v --tb=short
  ```

- [ ] Check coverage:
  ```bash
  pytest tests/ --cov=eol.rag_context --cov-report=term
  ```

### Results Tracking
- [ ] Document tests that now pass
- [ ] Document remaining failures
- [ ] Calculate new pass rate
- [ ] Verify >80% target achieved

### Cleanup
- [ ] Update TEST_FIX_PLAN.md with results
- [ ] Update .claude/context/testing.md with current status
- [ ] Commit fixes with clear messages
- [ ] Create PR if needed

---

## 📈 Progress Tracking

### Metrics
- [x] Starting: 15/52 passing (29%)
- [x] After Phase 2 (Priority 1): Actual 26/52 (50%) ✅ Better than expected!
- [x] After Phase 3 (Priority 2): Actual 26/52 (50%) - Fixed test structure issues
- [ ] After Phase 4: Expected 32/52 (62%)
- [ ] After Phase 5: Expected 38/52 (73%)
- [ ] After Phase 6: Expected 44/52 (85%)
- [ ] **Target: 42+/52 (>80%)**

### Time Tracking
- [x] Phase 1: ✅ Complete (30 min)
- [x] Phase 2: ✅ Complete (25 min)
- [x] Phase 3: ✅ Complete (20 min)
- [ ] Phase 4: ⏱️ Est. 15 min
- [ ] Phase 5: ⏱️ Est. 15 min
- [ ] Phase 6: ⏱️ Est. 30 min
- [ ] Phase 7: ⏱️ Est. 15 min
- [ ] **Total: ~3 hours**

---

## 🚨 Rollback Plan

If fixes cause regression:
1. [ ] Git stash or reset changes
2. [ ] Create feature branch for fixes
3. [ ] Implement fixes incrementally
4. [ ] Test each fix in isolation
5. [ ] Merge only stable fixes

---

## 📝 Notes

- Keep Redis container running throughout
- Stay in venv for all operations
- Test after each priority fix
- Don't break passing Redis tests
- Document any API changes
- Some tutorial tests may need test updates rather than code fixes

---

## ✅ Success Criteria

- [ ] Minimum 80% tests passing (42+ out of 52)
- [ ] All Redis integration tests remain green
- [ ] No regression in currently passing tests
- [ ] Clear documentation of changes
- [ ] All fixes committed with descriptive messages