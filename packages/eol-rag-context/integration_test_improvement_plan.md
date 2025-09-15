# Integration Test Coverage Improvement Plan

## Current Status

### Unit Test Coverage: ✅ **82.5%** (Exceeds 80% threshold!)

- Successfully improved from 71.78% to 82.5%
- document_processor.py at 82.67% (excellent coverage)
- All unit test thresholds met

### Integration Test Coverage: ⚠️ **59.1%** (Needs 60%)

- Currently 0.9% below the 60% threshold
- Local coverage: 52.95%
- CI coverage: 59.1%

## Coverage Analysis by Module

### Low Integration Test Coverage Modules

1. **batch_operations.py**: 0.00%
2. **main.py**: 0.00%
3. **document_processor.py**: 30.17%
4. **embeddings.py**: 33.18%
5. **file_watcher.py**: 34.07%
6. **server.py**: 38.52%

### Well-Covered Modules

- **async_task_manager.py**: 70.45%
- **config.py**: 75.49%
- **indexer.py**: 79.73%
- **knowledge_graph.py**: 86.45%

## Recommendations to Reach 60% Integration Test Coverage

### Quick Wins (High Impact, Low Effort)

1. **Add batch_operations integration test**
   - Currently at 0% coverage
   - Test batch document processing
   - Test streaming processor
   - Estimated coverage gain: +2-3%

2. **Enhance server.py integration tests**
   - Currently at 38.52% coverage
   - Add tests for MCP tools not currently tested
   - Test error handling paths
   - Estimated coverage gain: +1-2%

3. **Improve document_processor integration tests**
   - Currently at 30.17% coverage
   - Add tests for different file formats (PDF, DOCX, XML)
   - Test chunking strategies
   - Estimated coverage gain: +1-2%

### Implementation Strategy

1. **Focus on Critical Paths**
   - Test the most commonly used workflows
   - Ensure error handling is covered
   - Test with real Redis connections

2. **Use Existing Fixtures**
   - Leverage redis_store, embedding_manager fixtures
   - Use temp_test_directory for file operations
   - Reuse server_instance for MCP tool testing

3. **Test Real Integrations**
   - Test actual Redis operations (not mocked)
   - Test file system operations
   - Test concurrent operations

## Why Integration Tests Matter

Integration tests are failing at 59.1% because they test:

- Real Redis connections and operations
- Actual file system interactions
- Complete workflows across multiple components
- Error handling in production-like scenarios

## Next Steps

1. **Add Simple Integration Tests**

   ```python
   # Example: Test batch operations
   async def test_batch_processing(redis_store, document_processor):
       # Process multiple files in batch
       results = await batch_process_documents(files, redis_store)
       assert results.total > 0
   ```

2. **Enhance Existing Tests**
   - Add assertions to existing tests
   - Test error cases
   - Test edge conditions

3. **Fix Failing Tests**
   - test_list_indexing_tasks_tool is currently failing
   - Fix fixture compatibility issues
   - Ensure all tests run in CI environment

## Conclusion

We're extremely close to meeting all coverage thresholds:

- **Unit Tests**: ✅ 82.5% (exceeds 80% requirement)
- **Integration Tests**: 59.1% (just 0.9% below 60% requirement)

With minimal additional integration tests focusing on batch_operations and
server modules, we can easily exceed the 60% integration test threshold.
