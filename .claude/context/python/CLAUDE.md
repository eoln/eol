# Python Context Rules

## Code Standards

- Type hints for all functions
- Async/await for I/O
- Dataclasses for structures
- Abstract base classes for interfaces
- Comprehensive docstrings

## Testing Requirements

- Minimum 80% coverage
- Test success and failure paths
- Use pytest fixtures
- Mock external dependencies
- Test async code properly

## Code Style

- Follow PEP 8
- Use Black for formatting
- Use isort for imports
- Maximum line length: 88 (Black default)
- Meaningful variable names

## Error Handling

```python
# Custom exceptions
class RAGError(Exception):
    """Base exception for RAG operations"""
    pass

# Proper error handling
try:
    result = await operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    # Handle gracefully
```

## Virtual Environment

- ALWAYS use virtual environment
- Never install packages globally
- Prefer uv for package management
- Document dependencies clearly
