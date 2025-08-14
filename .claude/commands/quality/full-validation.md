# quality-full-validation - Complete Quality Gate Simulation

Comprehensive quality validation simulating the full CI/CD pipeline for the EOL RAG Framework.

## Purpose

Complete validation before PR submission or major milestone completion.

## Usage

```bash
/quality:full-validation [--strict]
```

## Complete Validation Suite

### 1. Code Quality (2-3 minutes)

```bash
# Format validation
python -m black src/ tests/ examples/ --check

# Import ordering
python -m isort src/ tests/ examples/ --check-only

# Comprehensive linting
python -m flake8 src/ tests/ --statistics --count

# Type checking
python -m mypy src/ --strict

# Docstring coverage
interrogate src/ -v --fail-under 80
```

### 2. Security Scanning (1-2 minutes)

```bash
# Dependency vulnerabilities
safety check

# Code security issues
bandit -r src/ -f json -o security-report.json

# Check for secrets
detect-secrets scan --baseline .secrets.baseline
```

### 3. Test Suite (3-5 minutes)

```bash
# Unit tests with coverage
python -m pytest tests/ \
  --cov=eol.rag_context \
  --cov-report=term \
  --cov-report=html \
  --cov-fail-under=80

# Integration tests (requires Redis)
python -m pytest tests/integration/ -v

# Performance benchmarks
python -m pytest tests/benchmarks/ --benchmark-only
```

### 4. RAG Performance Validation (2-3 minutes)

```python
# Indexing throughput
assert indexing_rate > 10  # docs/sec

# Search latency
assert search_p95 < 100  # ms

# Cache effectiveness
assert cache_hit_rate > 0.31  # 31%

# Memory usage
assert peak_memory < 2048  # MB
```

### 5. Documentation Check (1 minute)

```bash
# Build documentation
mkdocs build --strict

# Validate README links
markdown-link-check README.md

# API documentation generation
python -m pydoc -w src/eol/rag_context
```

## Quality Gate Summary

### Must Pass (Blocking)

- [ ] Black formatting compliant
- [ ] All tests passing
- [ ] Coverage >80%
- [ ] No high/critical security issues
- [ ] Type checking passes

### Should Pass (Warning)

- [ ] Performance targets met
- [ ] Documentation builds
- [ ] Complexity <10
- [ ] No deprecated patterns

### Nice to Have (Info)

- [ ] Coverage >90%
- [ ] Cache hit rate >35%
- [ ] Zero security warnings
- [ ] 100% docstring coverage

## Command Options

### Validation Modes

```bash
# Strict mode - fail on any warning
/quality:full-validation --strict

# CI mode - matches CI/CD pipeline exactly
/quality:full-validation --ci

# Local mode - skip CI-specific checks
/quality:full-validation --local
```

### Selective Validation

```bash
# Skip slow tests
/quality:full-validation --skip-integration

# Skip security scans
/quality:full-validation --skip-security

# Focus on performance
/quality:full-validation --performance-only
```

## Output Report

### Console Output

```
Quality Validation Report
========================
Code Quality:    ✅ PASSED
Security:        ✅ PASSED
Tests:           ✅ PASSED (87% coverage)
Performance:     ✅ PASSED
Documentation:   ⚠️  WARNING (85% complete)

Overall: PASSED with warnings
```

### Detailed Report

Generated at `quality-report.html` with:

- Violation details
- Coverage gaps
- Performance metrics
- Security findings
- Recommendations

## Success Criteria

- All blocking checks pass
- <5 warnings
- Performance within targets
- Documentation complete

## Time Estimate

- Quick Mode: 5-7 minutes
- Full Mode: 10-15 minutes
- Strict Mode: 15-20 minutes

## Integration with PRP

This validation is automatically run:

1. At end of each PRP phase
2. Before marking PRP complete
3. As part of PR creation
4. In CI/CD pipeline

## Troubleshooting

### Common Issues

1. **Import errors**: Check virtual environment activation
2. **Redis connection**: Ensure Redis is running
3. **Type errors**: Update type stubs
4. **Performance failures**: Check system resources

### Quick Fixes

```bash
# Auto-fix formatting
black src/ tests/ && isort src/ tests/

# Update dependencies
uv sync

# Reset Redis
redis-cli FLUSHDB
```

This comprehensive validation ensures code meets all quality standards before merging.
