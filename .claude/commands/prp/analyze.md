# prp-analyze - Extract Codebase Patterns and Context

Analyzes any codebase to extract implementation patterns, architectural decisions, and best practices for comprehensive context engineering in PRP development.

## Command Overview

**Purpose**: Extract and document existing implementation patterns from any codebase
**Target**: Any software project requiring pattern analysis for context engineering
**Output**: Pattern analysis saved to `.claude/context/[domain]/patterns.md`

## Usage

```bash
/prp:analyze [domain] [technology]
```

### Examples
```bash
# Analyze any domain patterns
/prp:analyze "authentication" "jwt-patterns"
/prp:analyze "data-processing" "pipeline-operations"
/prp:analyze "api-design" "error-handling"
/prp:analyze "database" "connection-pooling"
/prp:analyze "caching" "redis-operations"
```

## Analysis Process

### 1. Pattern Discovery Phase
```bash
# Search for domain-specific implementations
grep -r "class.*[Domain]" --include="*.py" src/
grep -r "async def.*[operation]" --include="*.py" src/
grep -r "[technology].*[pattern]" --include="*.py" src/
```

### 2. Code Structure Analysis
- **Language Patterns**: Type hints, async/await, design patterns, architectural decisions
- **Domain Components**: Core business logic, data processors, service layers
- **Infrastructure Operations**: Database patterns, caching strategies, external integrations
- **Testing Patterns**: Test frameworks, fixtures, mocks, integration strategies

### 3. Dependency Mapping
```bash
# Analyze package dependencies
uv tree
grep -h "^from\|^import" packages/*/src/**/*.py | sort | uniq
```

### 4. Performance Patterns
- Data processing throughput patterns
- Query optimization techniques
- Caching strategies and performance metrics
- Batch operation implementations
- Concurrency and parallelization patterns

## Output Format

### Generated Pattern Documentation
```markdown
# [Domain] Pattern Analysis

## Discovered Patterns

### Implementation Patterns
- **Data Processing**: [Processing strategies and transformation patterns found]
- **Service Integration**: [External service patterns and client implementations]
- **Storage Operations**: [Database access patterns and optimization techniques]
- **Caching Strategies**: [Cache implementations and performance patterns]

### Code Examples Found
```python
# Example from src/[domain]/processor.py
async def process_data(content: str, options: ProcessOptions) -> ProcessResult:
    """Actual implementation pattern from codebase"""
    ...
```

### Performance Optimizations
- **Batch Processing**: Pipeline patterns for data operations
- **Async Patterns**: Concurrent processing and I/O management
- **Connection Management**: Pool configuration, health checks, and resource optimization

### Testing Patterns
```python
# Example from tests/test_processor.py
@pytest.fixture
async def service_client():
    """Service fixture pattern for testing"""
    ...
```
```

## Analysis Targets

### Common Software Domains
1. **Data Processing**
   - Transformation strategies and pipelines
   - Validation and sanitization patterns
   - Content type handling

2. **Service Integration**
   - API client patterns and authentication
   - Error handling and retry mechanisms
   - Rate limiting and throttling

3. **Storage Layer**
   - Database access patterns
   - Caching strategies and invalidation
   - Data modeling and relationships

4. **Infrastructure Patterns**
   - Connection pooling and resource management
   - Configuration and environment handling
   - Monitoring and observability

### Language-Specific Best Practices
1. **Type Safety**
   - Type annotation patterns
   - Interface definitions
   - Generic programming patterns

2. **Concurrency Patterns**
   - Async/parallel operations
   - Resource management
   - Queue and worker patterns

3. **Testing Strategies**
   - Test organization patterns
   - Mock and stub strategies
   - Integration and end-to-end testing

## Integration Points

### Existing Context Files
The analysis enriches:
- `.claude/context/[domain]/` - Domain-specific patterns
- `.claude/context/[technology]/` - Technology-specific best practices
- `.claude/context/[language]/` - Language conventions and patterns

### Quality Standards
Ensures patterns follow:
- Performance targets specific to domain
- Test coverage requirements (>80%)
- Code quality standards (linting, type checking)
- Security and reliability best practices

## Command Options

### Scope Control
```bash
# Analyze specific package/module
/prp:analyze "authentication" "jwt-handling" --package=auth-service

# Deep analysis with examples
/prp:analyze "data-processing" "optimization" --deep

# Performance focus
/prp:analyze "caching" "performance" --metrics
```

### Output Options
```bash
# Save to specific location
/prp:analyze "database" "patterns" --output=.claude/context/database/patterns.md

# Generate comparison report
/prp:analyze "api-clients" "implementations" --compare
```

## Success Metrics

### Pattern Coverage
- [ ] All major domain components analyzed
- [ ] Technology patterns documented
- [ ] Language best practices captured
- [ ] Performance optimizations identified

### Documentation Quality
- [ ] Real code examples included
- [ ] Performance metrics documented
- [ ] Integration points mapped
- [ ] Testing patterns captured

## Next Steps

After analysis:
1. Use `/prp:create` to generate implementation plans
2. Reference patterns in development
3. Update context files with discoveries
4. Share patterns with team

This analysis command provides the foundation for context-aware software development by extracting and documenting proven patterns from any existing codebase, enabling comprehensive context engineering for PRP methodology.