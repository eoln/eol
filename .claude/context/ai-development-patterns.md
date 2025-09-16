# AI Development Patterns

## Context Verification Workflow

### Always Verify Before Acting

```python
# Pattern: Read, Verify, Act
async def safe_file_operation(file_path: str):
    # 1. Read current state
    if not await file_exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

    current_content = await read_file(file_path)

    # 2. Verify assumptions
    if not validate_content(current_content):
        raise ValueError("File content doesn't match expected format")

    # 3. Act with confidence
    return await process_file(file_path, current_content)
```

### Context Loading Strategy

1. **Check for local context first**: Look for CLAUDE.md in current directory
2. **Load domain context**: Pull in relevant domain knowledge
3. **Verify with codebase**: Check actual implementation
4. **Update context**: Document discoveries

## Error Recovery Patterns

### Graceful Degradation

```python
async def resilient_operation():
    try:
        # Primary approach
        return await primary_method()
    except PrimaryMethodError:
        try:
            # Fallback approach
            return await fallback_method()
        except FallbackError:
            # Graceful degradation
            return default_response()
```

### Error Context Preservation

```python
class ContextualError(Exception):
    def __init__(self, message: str, context: Dict[str, Any]):
        super().__init__(message)
        self.context = context

    def __str__(self):
        context_str = "\n".join(
            f"  {k}: {v}" for k, v in self.context.items()
        )
        return f"{super().__str__()}\nContext:\n{context_str}"
```

## Quality Gate Patterns

### Pre-commit Checks

```bash
# Run before allowing commits
async def quality_gate():
    checks = [
        ("Format", check_formatting),
        ("Lint", check_linting),
        ("Types", check_types),
        ("Tests", run_tests),
        ("Coverage", check_coverage)
    ]

    for name, check in checks:
        if not await check():
            raise QualityGateError(f"{name} check failed")
```

### Progressive Enhancement

1. Start with basic functionality
2. Add error handling
3. Implement retries
4. Add monitoring
5. Optimize performance

## Anti-Patterns to Avoid

### ❌ Assuming Without Verifying

```python
# BAD: Assumes file exists
content = read_file("config.json")

# GOOD: Verifies first
if file_exists("config.json"):
    content = read_file("config.json")
else:
    content = create_default_config()
```

### ❌ Silent Failures

```python
# BAD: Swallows errors
try:
    result = risky_operation()
except:
    pass

# GOOD: Handles explicitly
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    result = safe_default
```

### ❌ Overwriting Without Backup

```python
# BAD: Direct overwrite
write_file(path, new_content)

# GOOD: Backup first
backup_file(path)
write_file(path, new_content)
```

## Development Workflow

### Incremental Development

1. **Implement core functionality**
2. **Add tests**
3. **Handle errors**
4. **Optimize**
5. **Document**

### Test-Driven Approach

```python
# 1. Write test first
async def test_new_feature():
    result = await new_feature(input_data)
    assert result.status == "success"

# 2. Implement to pass test
async def new_feature(data):
    # Implementation
    return Result(status="success")

# 3. Refactor with confidence
async def new_feature(data):
    # Improved implementation
    validated = validate(data)
    processed = process(validated)
    return Result(status="success", data=processed)
```

## Context Window Management

### Prioritization Strategy

```python
class ContextManager:
    def __init__(self, max_tokens: int = 100000):
        self.max_tokens = max_tokens
        self.items = []

    def add(self, item: str, priority: int):
        self.items.append((priority, item))
        self._compress_if_needed()

    def _compress_if_needed(self):
        total = sum(len(item) for _, item in self.items)
        if total > self.max_tokens * 0.8:
            # Remove lowest priority items
            self.items.sort(reverse=True)
            while total > self.max_tokens * 0.7:
                self.items.pop()
                total = sum(len(item) for _, item in self.items)
```

## Monitoring and Observability

### Performance Tracking

```python
from functools import wraps
import time

def track_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise
    return wrapper
```

## Documentation Patterns

### Self-Documenting Code

```python
async def process_document(
    document: Document,
    chunk_size: int = 1000,
    overlap: int = 100
) -> ProcessedDocument:
    """
    Process document with configurable chunking.

    Args:
        document: Input document to process
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        ProcessedDocument with chunks and metadata

    Raises:
        ValueError: If chunk_size <= overlap
        ProcessingError: If document cannot be processed

    Example:
        >>> doc = Document("example.txt", "content")
        >>> result = await process_document(doc, chunk_size=500)
        >>> assert len(result.chunks) > 0
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    # Implementation
    ...
```

## Best Practices

1. Always verify before acting
2. Provide clear error messages
3. Implement graceful degradation
4. Track performance metrics
5. Document assumptions
6. Use type hints consistently
7. Handle edge cases explicitly
8. Keep context window efficient
