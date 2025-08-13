# Quality Gates

## Overview
Quality gates ensure code meets standards before merging. Based on `eol-rag-context-quality-gate.yml` GitHub Action.

## Code Quality Checks

### Formatting
```bash
# Black - Code formatter
python -m black src/ tests/ --check

# Fix automatically
python -m black src/ tests/
```

### Import Sorting
```bash
# isort - Import sorter
python -m isort src/ tests/ --check-only

# Fix automatically
python -m isort src/ tests/
```

### Linting
```bash
# Flake8 - Style guide enforcement
python -m flake8 src/ tests/

# Common issues:
# E501: Line too long (>88 chars)
# F401: Imported but unused
# E302: Expected 2 blank lines
```

### Type Checking
```bash
# Mypy - Static type checker
python -m mypy src/

# Strict mode
python -m mypy src/ --strict
```

## Testing Requirements

### Unit Tests
```bash
# Run all unit tests
python -m pytest tests/

# With coverage
python -m pytest tests/ --cov=eol.rag_context --cov-report=term

# Coverage threshold: 80%
python -m pytest tests/ --cov=eol.rag_context --cov-fail-under=80
```

### Integration Tests
```bash
# Requires Redis running
redis-server --daemonize yes

# Run integration tests
python -m pytest tests/integration/ -v

# With markers
python -m pytest -m integration
```

### Performance Tests
```bash
# Run benchmarks
python -m pytest tests/benchmarks/ --benchmark-only

# Performance thresholds:
# - Document indexing: >10 docs/sec
# - Vector search: <100ms latency
# - Cache hit rate: >31%
```

## Security Checks

### Dependency Scanning
```bash
# Safety - Check for known vulnerabilities
safety check

# Pip-audit alternative
pip-audit
```

### Code Security
```bash
# Bandit - Security linter
bandit -r src/

# Common issues:
# B101: assert_used
# B601: shell_injection
# B201: flask_debug_true
```

### Container Scanning
```bash
# Trivy - Container vulnerability scanner
trivy fs . --security-checks vuln,secret

# Scan Docker image
trivy image eol-rag-context:latest
```

## Multi-Version Testing

### Python Versions
```bash
# Test with Python 3.11
uv run --python 3.11 pytest tests/

# Test with Python 3.12
uv run --python 3.12 pytest tests/

# Use tox for matrix testing
tox -e py311,py312
```

## Performance Thresholds

### RAG Operations
| Operation | Threshold | Measurement |
|-----------|-----------|-------------|
| Document Indexing | >10 docs/sec | Throughput |
| Vector Search | <100ms | P95 latency |
| Embedding Generation | <50ms/chunk | Average time |
| Cache Hit Rate | >31% | Hit/(Hit+Miss) |
| Context Window | <80% | Usage percentage |

### Redis Operations
| Operation | Threshold | Measurement |
|-----------|-----------|-------------|
| Connection Pool | <50 connections | Max connections |
| Query Latency | <10ms | P95 latency |
| Pipeline Batch | >100 ops/batch | Average batch size |
| Memory Usage | <2GB | Peak memory |

## CI/CD Pipeline

### GitHub Actions Workflow
```yaml
name: Quality Gate
on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
    
    steps:
      # Code quality
      - run: black --check src/ tests/
      - run: isort --check-only src/ tests/
      - run: flake8 src/ tests/
      - run: mypy src/
      
      # Security
      - run: safety check
      - run: bandit -r src/
      
      # Tests with coverage
      - run: pytest --cov --cov-fail-under=80
      
      # Performance benchmarks
      - run: pytest tests/benchmarks/ --benchmark-only
```

## Local Quality Check

### Pre-commit Script
```bash
#!/bin/bash
# Run before committing

echo "Running quality checks..."

# Format check
python -m black --check src/ tests/ || exit 1

# Import sort check
python -m isort --check-only src/ tests/ || exit 1

# Lint
python -m flake8 src/ tests/ || exit 1

# Type check
python -m mypy src/ || exit 1

# Tests
python -m pytest tests/ -q || exit 1

echo "✅ All quality checks passed!"
```

## Quality Metrics Dashboard

### Key Metrics to Track
1. **Code Coverage**: Target >80%
2. **Technical Debt**: Keep low
3. **Cyclomatic Complexity**: <10 per function
4. **Duplication**: <5%
5. **Security Vulnerabilities**: 0 critical/high

### Monitoring Tools
- **Coverage.py**: Test coverage
- **Radon**: Complexity metrics
- **Vulture**: Dead code detection
- **PyDocStyle**: Docstring coverage

## Enforcement Strategy

### Required Checks (Blocking)
- [ ] Black formatting
- [ ] Tests passing
- [ ] Coverage >80%
- [ ] No security vulnerabilities

### Advisory Checks (Non-blocking)
- [ ] Documentation updated
- [ ] Performance benchmarks
- [ ] Complexity metrics

## Continuous Improvement

### Regular Reviews
- Weekly: Review failed checks
- Monthly: Update thresholds
- Quarterly: Tool evaluation

### Metrics Evolution
Track quality trends:
- Coverage trajectory
- Performance regression
- Security issue frequency
- Build success rate

## Best Practices
1. Run quality checks locally before push
2. Fix issues immediately, don't accumulate
3. Keep dependencies updated
4. Monitor quality metrics trends
5. Automate everything possible
6. Document quality standards
7. Train team on tools