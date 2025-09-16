# prp-validate - Validate Implementation Quality and Completeness

Comprehensive validation of PRP implementations against quality standards, performance targets, and software engineering best practices.

## Command Overview

**Purpose**: Validate that implementation meets all PRP requirements and quality standards
**Target**: Completed or in-progress PRP implementations
**Output**: Validation report with pass/fail status and recommendations

## Usage

```bash
/prp:validate [plan-file] [--full|--quick|--performance]
```

### Examples

```bash
# Full validation of completed implementation
/prp:validate .claude/plans/completed/prp-semantic-caching.md --full

# Quick validation during development
/prp:validate .claude/plans/pending/prp-vector-search.md --quick

# Performance-focused validation
/prp:validate .claude/plans/completed/prp-indexing.md --performance
```

## Validation Levels

### Quick Validation (--quick)

- Basic syntax checks
- Import verification
- Type hint presence
- Test existence

### Full Validation (--full)

- Complete quality gates
- Test coverage analysis
- Performance benchmarks
- Documentation completeness
- Integration testing

### Performance Validation (--performance)

- Throughput benchmarks
- Latency measurements
- Memory usage analysis
- Cache hit rates

## Validation Process

Follows the validation requirements from [`.claude/context/planning-methodology.md`](../../context/planning-methodology.md#success-metrics).

### 1. Code Quality Validation

#### Python Standards

```bash
# Format checking
black src/ tests/ --check

# Import ordering
isort src/ tests/ --check-only

# Linting
flake8 src/ tests/

# Type checking
mypy src/ --strict
```

#### Code Metrics

```python
# Complexity analysis
from radon.complexity import cc_visit
complexity = cc_visit(source_code)
assert all(f.complexity < 10 for f in complexity)
```

### 2. Test Coverage Validation

#### Coverage Requirements

```bash
# Run tests with coverage
pytest tests/ --cov=eol.rag_context --cov-report=term

# Validate coverage threshold
pytest tests/ --cov=eol.rag_context --cov-fail-under=80
```

#### Test Quality

```python
# Verify test patterns
- Async tests properly marked
- Fixtures used appropriately
- Mocks correctly implemented
- Edge cases covered
```

### 3. Domain Performance Validation

#### Document Processing

```python
async def validate_indexing_performance():
    """Validate indexing meets targets"""
    start = time.time()
    await indexer.index_documents(test_docs)
    rate = len(test_docs) / (time.time() - start)
    assert rate > 10, f"Indexing rate {rate} < 10 docs/sec"
```

#### Vector Search

```python
async def validate_search_latency():
    """Validate search performance"""
    latencies = []
    for query in test_queries:
        start = time.time()
        await redis_store.search_similar(query)
        latencies.append((time.time() - start) * 1000)

    p95 = np.percentile(latencies, 95)
    assert p95 < 100, f"P95 latency {p95}ms > 100ms"
```

#### Cache Performance

```python
def validate_cache_hit_rate():
    """Validate semantic cache effectiveness"""
    hits = cache.get_hit_count()
    misses = cache.get_miss_count()
    hit_rate = hits / (hits + misses)
    assert hit_rate > 0.31, f"Cache hit rate {hit_rate} < 31%"
```

### 4. Redis Integration Validation

#### Connection Health

```python
async def validate_redis_connection():
    """Validate Redis connectivity and pooling"""
    # Check connection pool
    assert redis_client.connection_pool.max_connections == 50

    # Test connectivity
    await redis_client.ping()

    # Verify indexes exist
    indexes = await redis_client.ft().list()
    assert "doc_index" in indexes
```

#### Vector Operations

```python
async def validate_vector_operations():
    """Validate vector index configuration"""
    info = await redis_client.ft("doc_index").info()

    # Check index configuration
    assert info["index_definition"]["algorithm"] == "HNSW"
    assert info["attributes"]["embedding"]["dim"] == 384
```

### 5. Documentation Validation

#### Code Documentation

```python
# Check docstring coverage
from interrogate import coverage
cov = coverage.get_coverage("src/")
assert cov > 90, f"Docstring coverage {cov}% < 90%"
```

#### API Documentation

```bash
# Validate API docs generation
python -m pydoc -w src/eol/rag_context
assert os.path.exists("rag_context.html")
```

## Validation Report Format

```markdown
# PRP Validation Report

## Summary
- **PRP**: prp-semantic-caching.md
- **Status**: ✅ PASSED (8/8 checks)
- **Date**: 2024-01-15
- **Confidence Score**: 9.5/10

## Quality Gates
| Check | Status | Details |
|-------|--------|---------|
| Code Formatting | ✅ | Black compliant |
| Import Ordering | ✅ | isort compliant |
| Type Checking | ✅ | mypy strict pass |
| Test Coverage | ✅ | 87% (target: 80%) |
| Performance | ✅ | All targets met |

## Performance Metrics
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Indexing Rate | >10 docs/s | 15.3 docs/s | ✅ |
| Search Latency | <100ms | 67ms (P95) | ✅ |
| Cache Hit Rate | >31% | 38.5% | ✅ |
| Memory Usage | <2GB | 1.3GB | ✅ |

## Test Results
- **Unit Tests**: 45/45 passed
- **Integration Tests**: 12/12 passed
- **Performance Tests**: 8/8 passed
- **Coverage**: 87% statements

## Recommendations
1. Consider adding more edge case tests
2. Optimize batch processing for larger datasets
3. Add monitoring for cache eviction rates

## Compliance
- [x] Follows domain patterns from context/
- [x] Uses Redis best practices
- [x] Implements Python async patterns
- [x] Meets performance targets
- [x] Documentation complete
```

## Validation Rules

### Critical (Must Pass)

- All tests passing
- Coverage >80%
- No security vulnerabilities
- Type checking passes

### Important (Should Pass)

- Performance targets met
- Documentation >90% coverage
- Complexity <10 per function
- No deprecated patterns

### Advisory (Nice to Have)

- Cache hit rate >35%
- Memory usage optimized
- Batch size tuning
- Advanced error recovery

## Command Options

### Validation Scope

```bash
# Validate specific phase
/prp:validate plan.md --phase=2

# Validate only changes
/prp:validate plan.md --changes-only

# Validate against baselines
/prp:validate plan.md --baseline=previous
```

### Output Options

```bash
# Generate detailed report
/prp:validate plan.md --report=detailed

# Output as JSON
/prp:validate plan.md --format=json

# Save report to file
/prp:validate plan.md --output=validation-report.md
```

## Success Criteria

### Implementation Complete

- [ ] All PRP tasks marked complete
- [ ] Quality gates passed
- [ ] Performance targets met
- [ ] Documentation updated

### Ready for Production

- [ ] Full validation passed
- [ ] Security scan clean
- [ ] Load testing successful
- [ ] Monitoring configured

This validation command ensures PRP implementations meet all quality standards and performance targets for production readiness.
