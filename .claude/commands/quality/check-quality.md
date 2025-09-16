# prp-check-quality - Comprehensive Quality Validation

## Command Overview

**Purpose**: Execute comprehensive quality checks for Python/RAG codebase
**Target**: EOL RAG Framework packages with Redis vector database
**Output**: Quality validation report with pass/fail status

## Usage

```bash
/prp:check-quality [package-name] [--full|--quick]
```

### Examples

```bash
# Full quality check for specific package
/prp:check-quality eol-rag-context --full

# Quick validation before commit
/prp:check-quality eol-rag-context --quick

# Check all packages
/prp:check-quality all
```

## Prerequisites

- Python virtual environment activated (`source .venv/bin/activate`)
- Redis running for integration tests
- All dependencies installed via `uv sync`

## Quality Gate Process

### 1. Code Formatting

```bash
# Check formatting
python -m black src/ tests/ --check

# Fix if needed
python -m black src/ tests/
```

### 2. Import Sorting

```bash
# Check import order
python -m isort src/ tests/ --check-only

# Fix if needed
python -m isort src/ tests/
```

### 3. Linting

```bash
# Run flake8
python -m flake8 src/ tests/

# Run mypy for type checking
python -m mypy src/
```

### 4. Security Checks

```bash
# Check for vulnerabilities
safety check

# Security linting
bandit -r src/
```

### 5. Run Tests with Coverage

```bash
# Run all tests
python -m pytest tests/ --cov=eol.rag_context --cov-report=term --cov-report=html

# Ensure coverage >80%
python -m pytest tests/ --cov=eol.rag_context --cov-fail-under=80
```

### 6. Integration Tests

```bash
# Start Redis if not running
redis-cli ping || redis-server --daemonize yes

# Run integration tests
python -m pytest tests/integration/ -v
```

## One-Line Quality Check

```bash
# Complete quality check
python -m black src/ tests/ --check && \
python -m isort src/ tests/ --check-only && \
python -m flake8 src/ tests/ && \
python -m mypy src/ && \
python -m pytest tests/ --cov=eol.rag_context --cov-fail-under=80
```

## Success Criteria

- ✅ All formatters pass without changes
- ✅ No linting errors
- ✅ No type errors
- ✅ All tests passing
- ✅ Coverage >80%
- ✅ No security vulnerabilities

## Troubleshooting

### Black Formatting Issues

```bash
# Auto-fix formatting
python -m black src/ tests/
```

### Import Order Issues

```bash
# Auto-fix imports
python -m isort src/ tests/
```

### Coverage Below Threshold

```bash
# View detailed coverage report
python -m pytest tests/ --cov=eol.rag_context --cov-report=html
# Open htmlcov/index.html in browser
```

### Type Checking Errors

```bash
# Run mypy with more detail
python -m mypy src/ --show-error-codes --pretty
```

## Notes

- Run this before every commit
- Fix issues immediately, don't accumulate technical debt
- If tests fail, check test output for specific failures
