# Documentation Update Command

## Purpose
Update and validate project documentation before PR submission.

## Prerequisites
- Code changes complete
- API changes identified
- Mkdocs installed (if using)

## Command Sequence

### 1. Update Code Documentation

#### Update Docstrings
```python
# Ensure all public functions have docstrings
# Example format:
def function_name(param: type) -> return_type:
    """
    Brief description.
    
    Args:
        param: Description of parameter
    
    Returns:
        Description of return value
    
    Raises:
        ExceptionType: When this occurs
    
    Example:
        >>> result = function_name(value)
        >>> assert result is not None
    """
```

#### Generate API Documentation
```bash
# Generate HTML documentation from docstrings
python -m pydoc -w src/eol/rag_context

# Or use Sphinx
sphinx-apidoc -o docs/api src/
sphinx-build -b html docs/ docs/_build
```

### 2. Update README

#### Check README Sections
- [ ] Installation instructions current
- [ ] Usage examples work
- [ ] API changes reflected
- [ ] Dependencies updated
- [ ] Badges accurate (CI status, coverage)

#### Validate README Links
```bash
# Check for broken links in Markdown
# Using markdown-link-check (npm install -g markdown-link-check)
markdown-link-check README.md

# Or using Python
python -m pytest tests/test_documentation.py::test_readme_links
```

### 3. Update CHANGELOG

#### Add Entry to CHANGELOG.md
```markdown
## [Unreleased]

### Added
- New feature description

### Changed
- Modified behavior description

### Fixed
- Bug fix description

### Deprecated
- Features to be removed

### Removed
- Deleted features

### Security
- Security fixes
```

### 4. Update MkDocs (if applicable)

#### Build and Validate
```bash
# Build documentation
mkdocs build --strict

# Serve locally to review
mkdocs serve --dev-addr=127.0.0.1:8000

# Check for errors
mkdocs build --strict 2>&1 | grep -E "WARNING|ERROR"
```

#### Update mkdocs.yml
```yaml
# Ensure new pages are included
nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - API Reference: api/
  - New Feature: new-feature.md  # Add new sections
```

### 5. Update Type Stubs (if needed)

```bash
# Generate type stubs
stubgen src/eol/rag_context -o stubs/

# Validate type stubs
mypy --strict stubs/
```

### 6. Update Examples

#### Validate Example Code
```bash
# Test all examples still work
for example in examples/*.py; do
    echo "Testing $example"
    python "$example" || exit 1
done
```

#### Update Example Documentation
```bash
# Ensure examples have clear comments
# Each example should include:
# - Purpose statement
# - Requirements
# - Expected output
# - Usage instructions
```

### 7. Documentation Checklist

Run through this checklist:
```bash
# Documentation validation checklist
echo "Documentation Checklist:"
echo "[ ] Docstrings updated for changed functions"
echo "[ ] README.md reflects current state"
echo "[ ] CHANGELOG.md entry added"
echo "[ ] Examples tested and working"
echo "[ ] API documentation generated"
echo "[ ] MkDocs builds without warnings"
echo "[ ] Type stubs updated (if applicable)"
echo "[ ] Migration guide (if breaking changes)"
```

## Success Criteria
- ✅ All public APIs documented
- ✅ Examples run successfully
- ✅ Documentation builds without errors
- ✅ Links validated
- ✅ CHANGELOG updated

## Troubleshooting

### MkDocs Build Errors
```bash
# Check for missing dependencies
pip install -r docs/requirements.txt

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('mkdocs.yml'))"
```

### Broken Example Code
```bash
# Run with verbose output
python -v examples/example.py

# Check imports
python -c "from eol.rag_context import *"
```

### Missing Docstrings
```bash
# Find functions without docstrings
pydocstyle src/ --select=D102,D103

# Or use interrogate
interrogate -v src/
```

## Documentation Standards

### Docstring Format (Google Style)
```python
def function(arg1: str, arg2: int = 0) -> bool:
    """Brief description of function.
    
    Longer description if needed, explaining the purpose
    and behavior of the function.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2, defaults to 0
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: If arg1 is empty
        TypeError: If arg2 is not numeric
    
    Example:
        >>> result = function("test", 42)
        >>> assert result is True
    
    Note:
        Additional notes about the function
    """
```

## Best Practices
1. Update docs with code changes
2. Include examples for new features
3. Keep README concise but complete
4. Version documentation with code
5. Test documentation code snippets