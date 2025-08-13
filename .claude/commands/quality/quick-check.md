# quality-quick-check - Fast Quality Validation

Rapid quality validation for Python/RAG code before commits or during active development.

## Purpose
Fast validation to catch common issues early in development cycle.

## Usage
```bash
/quality:quick-check [--fix]
```

## Command Sequence

### One-Line Quick Check
```bash
# Fast validation (no fixes)
python -m black src/ tests/ --check && \
python -m isort src/ tests/ --check-only && \
python -m flake8 src/ tests/ && \
python -m pytest tests/ -q --tb=no
```

### With Auto-Fix
```bash
# Auto-fix formatting issues
python -m black src/ tests/ && \
python -m isort src/ tests/ && \
python -m flake8 src/ tests/ && \
python -m pytest tests/ -q
```

## Validation Steps

### 1. Format Check (5 seconds)
```bash
black --check --diff src/ tests/
```

### 2. Import Order (3 seconds)
```bash
isort --check-only --diff src/ tests/
```

### 3. Basic Linting (10 seconds)
```bash
flake8 src/ tests/ --count --statistics
```

### 4. Quick Test Run (30 seconds)
```bash
pytest tests/ -q --tb=short -x
```

## Success Criteria
- ✅ Formatting correct (Black)
- ✅ Imports ordered (isort)
- ✅ No critical lint errors
- ✅ Core tests passing

## When to Use
- Before each commit
- After refactoring
- During active development
- Before running full validation

## Time Target
Total execution: <1 minute

## Skip Options
```bash
# Skip tests for ultra-fast check
/quality:quick-check --no-tests

# Skip format check if using auto-format
/quality:quick-check --no-format

# Fix issues automatically
/quality:quick-check --fix
```

This command provides rapid feedback during development without the overhead of full validation.