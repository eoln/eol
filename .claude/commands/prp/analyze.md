# prp-analyze - Extract RAG Patterns and Context

Analyzes the EOL codebase to extract RAG patterns, Redis implementations, and Python best practices for comprehensive context engineering.

## Command Overview

**Purpose**: Extract and document existing RAG implementation patterns
**Target**: Python packages with Redis vector database integration
**Output**: Pattern analysis saved to `.claude/context/[domain]/patterns.md`

## Usage

```bash
/prp:analyze [domain] [technology]
```

### Examples
```bash
# Analyze RAG-specific patterns
/prp:analyze "document-processing" "chunking-strategies"
/prp:analyze "vector-search" "redis-operations"
/prp:analyze "semantic-caching" "ttl-management"
/prp:analyze "embeddings" "model-providers"
```

## Analysis Process

### 1. Pattern Discovery Phase
```bash
# Search for domain-specific implementations
grep -r "class.*Chunk" --include="*.py" packages/
grep -r "async def.*embed" --include="*.py" packages/
grep -r "redis.*search" --include="*.py" packages/
```

### 2. Code Structure Analysis
- **Python Patterns**: Type hints, async/await, dataclasses
- **RAG Components**: Document processors, indexers, searchers
- **Redis Operations**: Vector indexes, pipelines, connection pools
- **Testing Patterns**: Pytest fixtures, mocks, async tests

### 3. Dependency Mapping
```bash
# Analyze package dependencies
uv tree
grep -h "^from\|^import" packages/*/src/**/*.py | sort | uniq
```

### 4. Performance Patterns
- Document indexing throughput patterns
- Vector search optimization techniques
- Caching strategies and hit rates
- Batch operation implementations

## Output Format

### Generated Pattern Documentation
```markdown
# [Domain] Pattern Analysis

## Discovered Patterns

### RAG Implementation Patterns
- **Document Processing**: [List of chunking strategies found]
- **Embedding Generation**: [Model providers and batch patterns]
- **Vector Operations**: [Redis vector search patterns]
- **Caching Strategies**: [Semantic cache implementations]

### Code Examples Found
```python
# Example from packages/eol-rag-context/src/document_processor.py
async def chunk_by_semantic(content: str, max_size: int = 1000) -> List[Chunk]:
    """Actual implementation pattern from codebase"""
    ...
```

### Performance Optimizations
- **Batch Processing**: Pipeline patterns for Redis operations
- **Async Patterns**: Concurrent document processing
- **Connection Management**: Pool configuration and health checks

### Testing Patterns
```python
# Example from tests/test_document_processor.py
@pytest.fixture
async def redis_store():
    """Redis fixture pattern for testing"""
    ...
```
```

## Analysis Targets

### RAG-Specific Domains
1. **Document Processing**
   - Chunking strategies (semantic, AST, fixed)
   - Metadata extraction
   - Content type detection

2. **Vector Operations**
   - Embedding generation patterns
   - Index design (HNSW vs FLAT)
   - Similarity search optimization

3. **Caching Layer**
   - Semantic similarity caching
   - TTL management strategies
   - Cache key generation

4. **Redis Patterns**
   - Connection pooling
   - Pipeline operations
   - Error recovery

### Python Best Practices
1. **Type Safety**
   - Type hint patterns
   - Protocol definitions
   - Generic types usage

2. **Async Patterns**
   - Concurrent operations
   - Rate limiting
   - Queue processing

3. **Testing Strategies**
   - Fixture patterns
   - Mock strategies
   - Integration testing

## Integration Points

### Existing Context Files
The analysis enriches:
- `.claude/context/rag/` - RAG-specific patterns
- `.claude/context/redis/` - Redis best practices
- `.claude/context/python/` - Python conventions

### Quality Standards
Ensures patterns follow:
- Performance targets (>10 docs/sec indexing)
- Test coverage requirements (>80%)
- Type safety standards (mypy compliance)

## Command Options

### Scope Control
```bash
# Analyze specific package
/prp:analyze "document-processing" "chunking" --package=eol-rag-context

# Deep analysis with examples
/prp:analyze "vector-search" "optimization" --deep

# Performance focus
/prp:analyze "caching" "performance" --metrics
```

### Output Options
```bash
# Save to specific location
/prp:analyze "redis" "patterns" --output=.claude/context/redis/patterns.md

# Generate comparison report
/prp:analyze "embeddings" "providers" --compare
```

## Success Metrics

### Pattern Coverage
- [ ] All major RAG components analyzed
- [ ] Redis patterns documented
- [ ] Python best practices captured
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

This analysis command provides the foundation for context-aware RAG development by extracting and documenting proven patterns from the existing codebase.