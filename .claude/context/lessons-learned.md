# Lessons Learned

## Common Pitfalls and Solutions

### Linting Issues

#### Black Formatting
**Issue**: Inconsistent code formatting
**Solution**: 
```bash
# Always run before commit
python -m black src/ tests/

# Check without modifying
python -m black --check src/ tests/
```

#### Import Ordering
**Issue**: Imports not properly sorted
**Solution**:
```bash
# Fix automatically
python -m isort src/ tests/

# Check without modifying
python -m isort --check-only src/ tests/
```

### User Correction Patterns

#### Over-Engineering
**Pattern**: Creating complex solutions for simple problems
**Correction**: Start with simplest solution that works, iterate if needed

#### Assuming Context
**Pattern**: Making changes without reading existing code
**Correction**: Always read file before editing, understand context

#### Ignoring Conventions
**Pattern**: Not following existing code style
**Correction**: Match the style of surrounding code

### Tool Usage Best Practices

#### Read Tool
- Always use before Edit tool
- Read enough context (not just target lines)
- Check for similar patterns in codebase

#### Edit Tool
- Preserve exact indentation
- Include enough context for unique match
- Use replace_all for systematic changes

#### Bash Tool
- Use virtual environment for Python commands
- Check command success/failure
- Avoid destructive commands without confirmation

## Testing Patterns

### Coverage Gaps
**Issue**: Tests missing edge cases
**Solution**: 
- Test happy path
- Test error conditions
- Test boundary values
- Test with None/empty inputs

### Async Testing
**Issue**: Improper async test setup
**Solution**:
```python
# Use pytest-asyncio
@pytest.mark.asyncio
async def test_async_function():
    result = await async_operation()
    assert result is not None
```

## Performance Optimizations

### Redis Connection Pooling
**Learning**: Individual connections slow down operations
**Solution**: Always use connection pooling
```python
pool = ConnectionPool(max_connections=50)
redis_client = Redis(connection_pool=pool)
```

### Batch Operations
**Learning**: Individual operations create overhead
**Solution**: Use pipelines for batch operations
```python
pipe = redis_client.pipeline()
for item in items:
    pipe.set(item.key, item.value)
await pipe.execute()
```

## Security Considerations

### Environment Variables
**Issue**: Hardcoded credentials in code
**Solution**: Always use environment variables
```python
import os
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not set")
```

### Input Validation
**Issue**: Trusting user input
**Solution**: Always validate and sanitize
```python
def process_input(user_input: str) -> str:
    # Validate
    if not user_input or len(user_input) > 1000:
        raise ValueError("Invalid input")
    
    # Sanitize
    cleaned = user_input.strip()
    return cleaned
```

## Documentation Gaps

### Missing Examples
**Issue**: Functions without usage examples
**Solution**: Include examples in docstrings
```python
def function():
    """
    Description here.
    
    Example:
        >>> result = function()
        >>> assert result is not None
    """
```

### Outdated README
**Issue**: README doesn't reflect current state
**Solution**: Update README with every major change

## CI/CD Issues

### Python Version Mismatches
**Issue**: Local Python version differs from CI
**Solution**: 
- Test with multiple Python versions locally
- Use `uv` for consistent environments
- Specify Python version in pyproject.toml

### Missing Dependencies
**Issue**: Works locally but fails in CI
**Solution**:
- Keep requirements.txt updated
- Use lock files for reproducible builds
- Test in clean environment

## Code Review Feedback

### Common Review Comments
1. "Add type hints" → Use type hints everywhere
2. "Handle this error" → Never ignore exceptions
3. "This could be async" → Use async for I/O operations
4. "Add test for this" → Maintain test coverage
5. "Document this behavior" → Keep docs current

## Workflow Improvements

### Plan Execution
**Learning**: Plans drift from reality
**Improvement**: Update plan checkboxes in real-time

### Context Management
**Learning**: Context gets stale
**Improvement**: Update context files immediately with discoveries

### Git Workflow
**Learning**: Large commits hard to review
**Improvement**: Make small, focused commits

## Anti-Patterns Discovered

### ❌ Batching Updates
Delaying status updates leads to confusion

### ❌ Skipping Tests
"Quick fixes" often break things

### ❌ Ignoring Warnings
Warnings often indicate real problems

### ❌ Copy-Paste Programming
Leads to maintenance nightmares

## Action Items from Lessons

1. **Create pre-commit hooks** for formatting/linting
2. **Document patterns** as they're discovered
3. **Update templates** with better defaults
4. **Add integration tests** for critical paths
5. **Monitor performance** continuously
6. **Review and update** context regularly

## Continuous Improvement

### Regular Reviews
- Weekly: Review recent issues
- Monthly: Update patterns and anti-patterns
- Quarterly: Major documentation updates

### Feedback Loop
1. Encounter issue
2. Document solution
3. Update relevant context
4. Share with team
5. Prevent recurrence