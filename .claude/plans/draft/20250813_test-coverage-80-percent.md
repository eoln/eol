# Test Coverage Improvement to 80% - Product Requirements Prompt

## Overview
Comprehensive plan to increase test coverage from 68.18% to 80% for both unit and integration tests in the eol-rag-context package, focusing on critical low-coverage modules and fixing existing test failures.

**Created**: 2025-08-13  
**Status**: Draft  
**Priority**: High  
**Estimated Duration**: 2 weeks  
**Scope**: Unit tests, integration tests, and test infrastructure improvements

## Implementation Confidence Score: 9/10
*Based on clear coverage gaps identified, existing test patterns, and straightforward implementation path*

## Research Summary

### Current Coverage Analysis
- **Current Coverage**: 68.18% (with Redis running)
- **Target Coverage**: 80% minimum
- **Gap to Close**: 11.82%
- **Failed Tests**: 32 failures, 10 errors
- **Total Test Files**: 11 unit test files, 5 integration test suites

### Critical Low Coverage Modules
| Module | Current Coverage | Lines to Cover | Priority |
|--------|-----------------|----------------|----------|
| `redis_client.py` | 46.39% | 85/185 | HIGH |
| `embeddings.py` | 48.15% | 63/139 | HIGH |
| `document_processor.py` | 54.31% | 124/305 | HIGH |
| `server.py` | 54.84% | 83/214 | HIGH |
| `file_watcher.py` | 55.69% | 116/304 | MEDIUM |
| `semantic_cache.py` | 71.04% | 40/171 | MEDIUM |

### Existing Test Patterns Found
```python
# Async test pattern from conftest.py
@pytest.fixture
async def redis_store(redis_config):
    store = RedisVectorStore(redis_config)
    await store.connect_async()
    yield store
    await store.cleanup()

# Mock pattern from test_embeddings.py  
@pytest.fixture
def mock_embedding_provider():
    provider = Mock(spec=EmbeddingProvider)
    provider.embed = AsyncMock(return_value=np.random.rand(384))
    return provider

# Parametrized test pattern
@pytest.mark.parametrize("chunk_size,overlap", [
    (100, 20),
    (500, 50),
    (1000, 100)
])
async def test_chunking_strategies(chunk_size, overlap):
    # Test implementation
```

### Key Dependencies Identified
- **Testing Framework**: pytest, pytest-asyncio, pytest-cov
- **Mocking**: unittest.mock, AsyncMock
- **Fixtures**: conftest.py with shared fixtures
- **Redis Testing**: Redis Stack Server required for integration tests
- **Coverage Tools**: pytest-cov with branch coverage

## Architecture Overview

### Test Structure (Directory-Specific Fixtures)
```
tests/
├── conftest.py              # MINIMAL shared utilities only
│   └── temp_dir             # Safe to share (no I/O)
├── unit/                    # Unit tests (fast, isolated)
│   ├── conftest.py          # Unit-specific fixtures (MOCKED)
│   │   ├── mock_redis_store
│   │   ├── mock_embedding_provider
│   │   ├── mock_mcp_server
│   │   └── unit_test_config
│   ├── test_config.py
│   ├── test_embeddings.py
│   ├── test_document_processor.py
│   ├── test_indexer.py
│   ├── test_redis_client.py
│   ├── test_semantic_cache.py
│   ├── test_knowledge_graph.py
│   ├── test_file_watcher.py
│   ├── test_mcp_server.py
│   └── test_server.py
└── integration/             # Integration tests (real dependencies)
    ├── conftest.py          # Integration-specific fixtures (REAL)
    │   ├── redis_stack_server
    │   ├── real_redis_store
    │   ├── integration_config
    │   └── cleanup_redis_data
    ├── test_redis_integration.py
    ├── test_indexing_integration.py
    ├── test_full_workflow_integration.py
    └── test_tutorial_examples.py
```

### Fixture Isolation Strategy
- **Root conftest.py**: Only truly shared, non-I/O utilities
- **Unit conftest.py**: Fast, mocked, deterministic fixtures
- **Integration conftest.py**: Real services, slower, environmental
- **NO cross-contamination** between test types
- **Clear scoping**: fixtures stay in their test domain

## Implementation Tasks

### Phase 1: Fix Failing Tests (Days 1-2)
- [ ] **Standardize on Python 3.13 across project and CI**
  ```bash
  # Update .pre-commit-config.yaml to use Python 3.13
  # Change language_version from python3.11 to python3.13
  sed -i 's/language_version: python3.11/language_version: python3.13/g' .pre-commit-config.yaml
  
  # Verify the change
  grep "language_version" .pre-commit-config.yaml
  
  # Update GitHub Actions workflow to include Python 3.13 in test matrix
  # Edit .github/workflows/test.yml to add python-version: ["3.11", "3.12", "3.13"]
  
  # Clean and reinstall hooks
  pre-commit clean
  pre-commit install --install-hooks
  pre-commit run --all-files  # Test all hooks work
  
  # Verify Python version consistency
  python --version  # Should show 3.13.x
  ```

- [ ] **Update GitHub Actions workflow for Python 3.13 matrix**
  ```yaml
  # Update .github/workflows/test.yml (or create if missing)
  # Add Python 3.13 to test matrix: ["3.11", "3.12", "3.13"]
  # Ensure Redis Stack installation works across all versions
  # Use uv for faster dependency installation
  # Only upload coverage report once (from Python 3.13)
  ```

- [ ] **Reorganize test structure with proper fixture separation**
  ```bash
  # Create unit test directory and move existing unit tests
  mkdir -p tests/unit
  mv tests/test_*.py tests/unit/
  
  # Keep integration tests in their directory
  # tests/integration/ already exists with proper conftest.py
  ```

- [ ] **Fix file_watcher fixture issues**
  ```python
  # Fix mock configuration in tests/test_file_watcher.py
  @pytest.fixture
  def handler(mock_watcher):
      mock_watcher = Mock(spec=FileWatcher)  # Remove side_effect
      source_path = Path("/test/path")
      source_id = "test_source"
      file_patterns = ["*.py", "*.md"]
      return FileChangeHandler(mock_watcher, source_path, source_id, file_patterns)
  ```

- [ ] **Fix MCP server async mock issues**
  ```python
  # Properly mock async MCP server
  @pytest.fixture
  async def mock_mcp_server():
      server = AsyncMock(spec=MCPServer)
      server.initialize = AsyncMock()
      server.index_directory = AsyncMock(return_value={"status": "success"})
      return server
  ```

- [ ] **Fix semantic cache Redis mocks**
  ```python
  # tests/conftest.py - Unit test fixtures (MOCKED)
  @pytest.fixture
  async def mock_redis_for_cache():
      redis_mock = AsyncMock()
      redis_mock.get = AsyncMock(return_value=None)
      redis_mock.set = AsyncMock(return_value=True)
      redis_mock.zadd = AsyncMock(return_value=1)
      return redis_mock
  ```

- [ ] **Create dedicated unit test fixtures directory**
  ```python
  # tests/conftest.py - MINIMAL shared utilities only
  @pytest.fixture
  def temp_dir() -> Generator[Path, None, None]:
      """Safe to share - just creates temp directory."""
      temp_path = Path(tempfile.mkdtemp())
      yield temp_path
      shutil.rmtree(temp_path)
  
  # tests/unit/conftest.py - Unit-specific fixtures (MOCKED)
  @pytest.fixture
  def mock_redis_store():
      store = Mock(spec=RedisVectorStore)
      store.connect_async = AsyncMock()
      store.search_similar = AsyncMock(return_value=[])
      store.vector_search = AsyncMock(return_value=[])
      return store
      
  @pytest.fixture
  def unit_test_config(temp_dir: Path):
      """Unit test config with cache disabled."""
      config = RAGConfig()
      config.data_dir = temp_dir / "data"
      config.cache.enabled = False  # No real I/O for units
      return config
  
  # tests/integration/conftest.py - Integration-specific fixtures (REAL)
  @pytest.fixture
  async def real_redis_store(redis_config):
      store = RedisVectorStore(redis_config)
      await store.connect_async()
      yield store
      await store.cleanup()
      await store.disconnect()
  ```

### Phase 2: High Priority Module Tests (Days 3-5)

#### redis_client.py (+20% coverage needed)
- [ ] **Test connection management**
  ```python
  async def test_connection_pooling():
      store = RedisVectorStore(config)
      await store.connect_async()
      assert store.pool.max_connections == 50
      await store.disconnect()
  
  async def test_connection_retry():
      store = RedisVectorStore(bad_config)
      with pytest.raises(ConnectionError):
          await store.connect_async()
  ```

- [ ] **Test vector operations**
  ```python
  async def test_vector_index_creation():
      await store.create_index("test_index", dimension=384)
      info = await store.get_index_info("test_index")
      assert info["dimension"] == 384
  
  async def test_vector_search():
      embedding = np.random.rand(384)
      results = await store.vector_search(embedding, k=5)
      assert len(results) <= 5
  ```

- [ ] **Test batch operations**
  ```python
  async def test_batch_insert():
      documents = [create_test_doc(i) for i in range(100)]
      result = await store.batch_insert(documents)
      assert result.success_count == 100
  
  async def test_pipeline_operations():
      async with store.pipeline() as pipe:
          for i in range(10):
              pipe.hset(f"doc:{i}", mapping={"content": f"test {i}"})
          results = await pipe.execute()
      assert len(results) == 10
  ```

#### embeddings.py (+20% coverage needed)
- [ ] **Test all provider types**
  ```python
  @pytest.mark.parametrize("provider_type", ["openai", "anthropic", "local", "huggingface"])
  async def test_embedding_providers(provider_type):
      provider = EmbeddingManager.create_provider(provider_type)
      embedding = await provider.embed("test text")
      assert embedding.shape == (384,)
  
  async def test_openai_provider_with_mock():
      with patch("openai.AsyncClient") as mock_client:
          mock_client.embeddings.create = AsyncMock(
              return_value=Mock(data=[Mock(embedding=[0.1]*1536)])
          )
          provider = OpenAIProvider()
          result = await provider.embed("test")
          assert len(result) == 1536
  ```

- [ ] **Test batch processing**
  ```python
  async def test_batch_embeddings():
      texts = ["text1", "text2", "text3"]
      embeddings = await manager.embed_batch(texts, batch_size=2)
      assert len(embeddings) == 3
      assert all(e.shape == (384,) for e in embeddings)
  
  async def test_embedding_cache():
      text = "cached text"
      emb1 = await manager.embed(text)
      emb2 = await manager.embed(text)  # Should hit cache
      assert np.array_equal(emb1, emb2)
  ```

#### document_processor.py (+15% coverage needed)
- [ ] **Test all chunking strategies**
  ```python
  @pytest.mark.parametrize("strategy,expected_chunks", [
      ("fixed", 10),
      ("semantic", 5),
      ("ast", 8),
      ("markdown", 7),
      ("sliding_window", 12)
  ])
  async def test_chunking_strategies(strategy, expected_chunks):
      processor = DocumentProcessor(chunk_strategy=strategy)
      content = load_test_content(f"{strategy}_test.txt")
      chunks = await processor.chunk_content(content)
      assert len(chunks) == expected_chunks
  ```

- [ ] **Test file format processing**
  ```python
  @pytest.mark.parametrize("file_type,processor_method", [
      ("test.json", "process_json"),
      ("test.yaml", "process_yaml"),
      ("test.md", "process_markdown"),
      ("test.py", "process_code"),
      ("test.pdf", "process_pdf")
  ])
  async def test_file_processors(file_type, processor_method):
      processor = DocumentProcessor()
      result = await processor.process_file(f"test_data/{file_type}")
      assert result.chunks is not None
      assert result.metadata["file_type"] == file_type.split(".")[-1]
  ```

### Phase 3: Medium Priority Tests (Days 6-7)

#### server.py (+15% coverage needed)
- [ ] **Test all MCP tools**
  ```python
  async def test_index_directory_tool():
      server = MCPServer()
      result = await server.tools["index_directory"]({
          "path": "/test/path",
          "recursive": True
      })
      assert result["status"] == "success"
  
  async def test_search_context_tool():
      result = await server.tools["search_context"]({
          "query": "test query",
          "k": 5
      })
      assert len(result["results"]) <= 5
  ```

- [ ] **Test resource handlers**
  ```python
  async def test_context_resource():
      resource = await server.get_resource("context://current")
      assert resource["type"] == "context"
      assert "content" in resource
  
  async def test_stats_resource():
      stats = await server.get_resource("stats://indexing")
      assert "documents_indexed" in stats
      assert "total_chunks" in stats
  ```

#### file_watcher.py (+10% coverage needed)
- [ ] **Test file system events**
  ```python
  async def test_file_created_event():
      handler = FileChangeHandler(watcher, path, source_id)
      event = FileCreatedEvent("/test/new_file.py")
      await handler.on_created(event)
      assert watcher.index_file.called
  
  async def test_file_modified_event():
      event = FileModifiedEvent("/test/modified.py")
      await handler.on_modified(event)
      assert watcher.reindex_file.called
  ```

### Phase 4: Integration Test Improvements (Days 8-9)

- [ ] **Ensure Redis Stack is available**
  ```python
  # integration/conftest.py
  @pytest.fixture(scope="session")
  async def redis_stack():
      """Ensure Redis Stack is running for integration tests."""
      if not check_redis_stack():
          pytest.skip("Redis Stack not available")
      yield
      # Cleanup after all tests
  ```

- [ ] **Add retry logic for flaky tests**
  ```python
  @pytest.mark.flaky(retries=3, delay=1)
  async def test_concurrent_operations():
      # Test implementation
  ```

- [ ] **Create test data generators**
  ```python
  @pytest.fixture
  def test_documents():
      return [
          Document(id=f"doc_{i}", content=f"Test content {i}")
          for i in range(100)
      ]
  ```

### Phase 5: Coverage Validation & Documentation (Days 10)

- [ ] **Run coverage analysis**
  ```bash
  # Run full test suite with coverage
  pytest tests/ --cov=eol.rag_context \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-branch \
    --cov-fail-under=80
  ```

- [ ] **Document test patterns**
  ```markdown
  # Testing Guidelines
  
  ## Unit Tests
  - Mock all external dependencies
  - Test edge cases and error conditions
  - Use parametrized tests for multiple scenarios
  
  ## Integration Tests
  - Require Redis Stack Server
  - Test end-to-end workflows
  - Validate performance requirements
  ```

## Quality Gates

### Test Quality Metrics
```bash
# Lint test code
flake8 tests/ --max-line-length=100

# Type check tests
mypy tests/ --ignore-missing-imports

# Check test coverage
pytest tests/ --cov=eol.rag_context --cov-fail-under=80

# Generate coverage report
pytest tests/ --cov-report=html
open coverage/html/index.html
```

### Performance Validation
```python
# Benchmark critical operations
@pytest.mark.benchmark
async def test_indexing_performance(benchmark):
    result = await benchmark(index_documents, test_docs)
    assert result.avg_time < 0.1  # <100ms per doc

@pytest.mark.benchmark  
async def test_search_latency(benchmark):
    result = await benchmark(search_similar, "query")
    assert result.avg_time < 0.1  # <100ms
```

### Continuous Integration
```yaml
# .github/workflows/test.yml updates - Python 3.13 matrix support
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]  # Add 3.13 support
        
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install Redis Stack
      run: |
        curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
        echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
        sudo apt-get update
        sudo apt-get install redis-stack-server
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip uv
        uv pip install -e ".[test]"
        
    - name: Start Redis Stack
      run: redis-stack-server --daemonize yes
      
    - name: Run tests with coverage
      run: |
        pytest tests/ --cov=eol.rag_context --cov-fail-under=80 --cov-report=xml
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      if: matrix.python-version == '3.13'  # Only upload once
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

## Success Metrics

### Coverage Targets
- [x] Overall coverage ≥ 80%
- [x] Branch coverage ≥ 70%
- [x] Critical modules ≥ 75% coverage
- [x] No untested public APIs

### Test Quality
- [x] All tests passing (0 failures)
- [x] No flaky tests
- [x] Test execution < 60 seconds
- [x] Clear test names and documentation

### Module-Specific Targets
| Module | Target Coverage | Critical Functions Covered |
|--------|----------------|---------------------------|
| `redis_client.py` | 75% | connect, search, index |
| `embeddings.py` | 75% | all providers, batch |
| `document_processor.py` | 75% | all chunking strategies |
| `server.py` | 70% | all MCP tools |
| `file_watcher.py` | 70% | all event handlers |

## Risk Mitigation

### Technical Risks
- **Python Version Standardization**: CRITICAL - Standardize on Python 3.13 across pre-commit, CI, and development
- **CI Matrix Compatibility**: Ensure tests pass on Python 3.11, 3.12, and 3.13
- **Redis Dependency**: Create comprehensive mocks for unit tests
- **Async Complexity**: Use pytest-asyncio properly, avoid event loop issues
- **Mock Configuration**: Ensure mocks match actual interfaces
- **Test Isolation**: Clean up after each test to prevent interference

### Process Risks
- **Time Constraints**: Focus on high-impact modules first
- **Flaky Tests**: Add retry logic and proper timeouts
- **Environment Issues**: Document Redis setup clearly
- **Coverage Accuracy**: Use branch coverage for better metrics

## Implementation Timeline

### Week 1 (Days 1-5)
- Day 1-2: Fix all failing tests
- Day 3-5: Implement high-priority module tests

### Week 2 (Days 6-10)
- Day 6-7: Complete medium-priority tests
- Day 8-9: Improve integration tests
- Day 10: Validate coverage and document

## References

### Internal Documentation
- `.claude/findings/20250813_test_coverage_analysis.md` - Coverage gap analysis
- `.claude/findings/20250813_redis_test_setup_guide.md` - Redis setup guide
- `tests/integration/TESTING_GUIDE.md` - Integration test guide
- `.claude/context/python/testing-strategies.md` - Python test patterns

### External Resources
- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio guide](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Redis Stack testing](https://redis.io/docs/stack/testing/)

## Task Dependencies

### Critical First Tasks (MUST Complete Before All Others)
- Standardize on Python 3.13 across project and CI
- Update GitHub Actions workflow for Python 3.13 matrix

### Independent Tasks (Can Start After Python Standardization)
- Reorganize test structure with proper fixture separation
- Fix file_watcher fixture issues
- Fix MCP server async mock issues  
- Fix semantic cache Redis mocks
- Document test patterns

### Sequential Dependencies
- Test connection management → Test vector operations → Test batch operations (redis_client)
- Test all provider types → Test batch processing (embeddings)
- Test all chunking strategies → Test file format processing (document_processor)

### Parallel Execution Opportunities
- Phase 1 fixes can be done in parallel
- Unit tests for different modules can be developed concurrently
- Documentation can proceed alongside test development

## Git Branch Strategy

### Execution Plan
1. **Plan Approval**: Move from draft/ to ready/
2. **Branch Creation**: `git checkout -b feat/test-coverage-80`
3. **Plan Activation**: Move to pending/
4. **Incremental Commits**:
   - Commit after each phase completion
   - Push daily for backup
   - Use descriptive commit messages
5. **PR Creation**: When coverage reaches 80%
6. **Plan Completion**: Move to completed/ after PR merge

## Progress Tracking Template

```markdown
## Progress Update [Date]
- Completed: X of 13 major tasks
- Current Coverage: XX.XX%
- Current Focus: [Active task]
- Blockers: [Any issues]
- Next Steps: [Upcoming tasks]
```

## Lessons Learned Template

```markdown
## Lessons Learned

### What Worked Well
- [To be filled during execution]

### What Could Be Improved  
- [To be filled during execution]

### New Patterns Discovered
- [To be filled during execution]

### Action Items
- [ ] Update `.claude/context/python/testing-strategies.md`
- [ ] Document new test fixtures
- [ ] Create reusable test utilities
```

## Next Steps

1. Review and approve this PRP
2. Move plan from draft/ to ready/
3. Create feature branch: `feat/test-coverage-80`
4. Move plan to pending/
5. Begin Phase 1: Fix failing tests
6. Daily progress updates using template above
7. Final validation against 80% target
8. Create PR and move to completed/

---

*This PRP provides a comprehensive roadmap to achieve 80% test coverage with confidence score 9/10.*

*Generated with context from eol-rag-context codebase analysis and planning-methodology.md*