# EOL RAG Framework - Project Context for AI Assistants

## Overview
EOL is a Retrieval-Augmented Generation (RAG) framework for building intelligent, context-aware AI applications. The framework provides production-ready tools for document indexing, semantic search, and dynamic context retrieval using Redis as a high-performance vector database. The `eol-rag-context` MCP server is the core component that enables applications to leverage RAG capabilities through the Model Context Protocol.

## Key Components

### eol-rag-context MCP Server
The core RAG service that provides:
- **Document Processing**: Intelligent chunking and hierarchical indexing (concepts → sections → chunks)
- **Vector Search**: High-performance semantic search using Redis vector database
- **Semantic Caching**: Reduces LLM API calls with intelligent response caching (31% hit rate target)
- **Knowledge Graph**: Builds entity relationships for enhanced context understanding
- **MCP Integration**: Serves RAG capabilities to any MCP-compatible client
- **File Watching**: Auto-indexes changes for always-current context

## Core Principles

### 1. Context-First Development
- Leverage Redis v8 vector database for context storage
- Implement semantic caching (target: 31% hit rate)
- Use content-aware chunking (AST for code, semantic for text)
- Manage LLM context window efficiently

### 2. RAG-First Architecture
- Every component designed for optimal RAG performance
- Hierarchical document structure for multi-level retrieval
- Content-aware processing (AST for code, semantic for text)
- Vector embeddings optimized for semantic similarity
- Caching layer to minimize redundant computations

## Technical Stack

### Core Technologies
- **Language**: Python 3.11+
- **Vector Database**: Redis v8 with vector search
- **LLM Integration**: Multi-provider (Anthropic, OpenAI, local)
- **Package Management**: uv (not pip or poetry)
- **License**: GPL-3.0

### Key Libraries
- **Redis**: `redis[vector]`, `redisvl` for vector operations
- **CLI**: `typer`, `rich` for terminal interface
- **MCP**: `fastmcp` for Model Context Protocol
- **Testing**: `pytest`, `pytest-asyncio`

## Nested CLAUDE.md Architecture

The `.claude/` folder uses nested CLAUDE.md files for directory-specific behavior:
- Each subdirectory may contain its own CLAUDE.md with local rules
- Subdirectory rules override parent rules for that context
- Always check for local CLAUDE.md when entering directories

## Context Engineering

Four strategies for effective AI assistance:
- **WRITE**: Persistent memory via `.claude/plans/`
- **SELECT**: Dynamic context loading from `.claude/context/`
- **COMPRESS**: Optimize context window usage (<80%)
- **ISOLATE**: Respect directory and security boundaries

## Planning Methodology

- Use structured plans in `.claude/plans/`
- Follow state lifecycle: draft → ready → pending → completed
- ALWAYS create dedicated git branch for plan execution (e.g., feat/plan-name)
- Update task status immediately (never batch)
- Mark completions in real-time
- Create PR when plan moves to completed state

## Command Patterns

- Load commands from `.claude/commands/` for common workflows
- PRP (Product Requirements Prompt) commands for implementation blueprints
- Quality commands for validation and checks
- Commands include prerequisites and success criteria

## Python Environment

- ALWAYS activate virtual environment before running scripts
- Use `source .venv/bin/activate` or appropriate venv activation
- Prefer `uv` for package management and environment handling
- Never install packages globally - always use venv

## Project Structure

```
eol/
├── packages/          # Monorepo packages (uv workspace)
│   ├── eol-rag-context/ # Core RAG implementation with MCP server
│   │   ├── src/         # Source code
│   │   ├── tests/       # Unit and integration tests
│   │   └── examples/    # Usage examples
│   ├── eol-core/       # Core framework utilities (planned)
│   ├── eol-cli/        # CLI interface (planned)
│   └── eol-sdk/        # Python SDK for RAG apps (planned)
├── examples/           # Example RAG applications
└── .claude/            # AI assistant context and commands
    ├── CLAUDE.md       # Directory-specific rules
    ├── context/        # Domain-specific knowledge
    ├── plans/          # Task planning with states
    └── commands/       # Reusable command patterns
```

## Code Style and Conventions

### Python Code
- Use type hints for all function signatures
- Async/await for I/O operations
- Dataclasses for data structures
- Abstract base classes for interfaces
- Comprehensive docstrings with examples

### File Naming
- Python modules: lowercase with underscores
- Classes: PascalCase
- Functions/variables: snake_case

### Error Handling
- Custom exception classes for each error type
- Always provide context in error messages
- Implement retry logic with exponential backoff
- Use circuit breakers for external services

## Implementation Guidelines

### Building RAG Applications

1. **Document Processing Pipeline**
   - Use appropriate chunking strategies for content type
   - Generate embeddings with consistent models (all-MiniLM-L6-v2)
   - Store metadata for filtering and ranking

2. **Vector Search Optimization**
   - Index documents at multiple hierarchy levels
   - Use hybrid search (vector + keyword) when appropriate
   - Implement result re-ranking based on relevance

### When Working with RAG

1. **Use Appropriate Chunking**
   - Code: AST-based chunking
   - Text: Semantic paragraph chunking
   - Markdown: Section-based chunking

2. **Implement Caching**
   - Semantic similarity for cache hits
   - TTL-based expiration
   - Track cache performance metrics

3. **Optimize Context Window**
   - Prioritize relevant context
   - Compress when > 80% full
   - Remove low-relevance items first

### When Integrating with Redis

```python
# Always use connection pooling
redis_client = Redis(connection_pool=pool)

# Use pipeline for batch operations
pipe = redis_client.pipeline()
for item in items:
    pipe.hset(...)
pipe.execute()

# Implement proper error handling
try:
    result = await redis_client.get(key)
except RedisError as e:
    # Handle gracefully, try fallback
```

## Testing Requirements

### Test Coverage
- Minimum 80% code coverage
- All public APIs must have tests
- Both unit and integration tests required

### Test Patterns
```python
# Use fixtures for setup
@pytest.fixture
async def setup():
    # Setup code
    yield resources
    # Teardown code

# Test both success and failure cases
async def test_operation_success(setup):
    # Test happy path

async def test_operation_failure(setup):
    # Test error handling
```

## Security Considerations

- **Never hardcode credentials** - Use environment variables
- **Validate all inputs** - For all external data
- **Sanitize LLM outputs** - Before execution
- **Implement rate limiting** - For all external APIs
- **Use least privilege** - For all service accounts

## Performance Targets

- **Document Indexing**: > 10 documents/second
- **Vector Search Latency**: < 100ms for 10k documents
- **Cache Hit Rate**: > 31% for semantic cache
- **Embedding Generation**: < 50ms per chunk
- **Context Window Usage**: Keep below 80% normally
- **MCP Response Time**: < 500ms for typical queries

## Common Patterns

### RAG Pipeline Implementation
```python
# Initialize RAG components
indexer = DocumentIndexer(redis_store, embedding_manager)
cache = SemanticCache(redis_store, embedding_manager)

# Index documents
await indexer.index_folder("./docs")

# Perform RAG search
query = "How does authentication work?"
context = await redis_store.search_similar(query, k=5)
response = await llm.generate(prompt=query, context=context)
```

### Health Checking
```python
# Implement health checks for all services
async def health_check() -> Dict[str, Any]:
    return {
        'status': 'healthy',
        'checks': {
            'redis': await check_redis(),
            'models': await check_models()
        }
    }
```

## Debugging Tips

1. **Enable verbose logging**: `eol run --verbose`
2. **Check dependency graph**: `eol deps graph`
3. **Monitor Redis**: Use RedisInsight on port 8001
4. **Track context usage**: Check window status regularly
5. **Profile performance**: Use `cProfile` for bottlenecks

## Do's and Don'ts

### DO:
- ✅ Write comprehensive docstrings
- ✅ Handle errors gracefully
- ✅ Use type hints everywhere
- ✅ Test edge cases
- ✅ Document dependencies
- ✅ Optimize for context window
- ✅ Use semantic caching

### DON'T:
- ❌ Hardcode credentials
- ❌ Skip error handling
- ❌ Exceed context window limits
- ❌ Use synchronous I/O
- ❌ Forget dependency resolution
- ❌ Skip tests

## Getting Help

- Check `.claude/context/` for detailed documentation
- Review `examples/` for reference implementations
- Use `eol --help` for CLI documentation
- Refer to test files for usage examples

## CI/CD Best Practices

### Always Check PR Status After Pushing
**IMPORTANT**: After pushing any changes, immediately check PR checks status to discover CI/CD failures ASAP:
```bash
gh pr checks <PR_NUMBER>  # Check status of all checks
gh run list --branch <BRANCH>  # List recent workflow runs
gh run view <RUN_ID> --log-failed  # View failed job logs
```

This allows you to:
- Fix issues quickly before they block merges
- Catch environment-specific problems (e.g., Python version differences)
- Ensure tests pass in the CI environment, not just locally

## Contributing

When contributing to EOL:
1. Follow the code style guide
2. Write tests for new features
3. Update documentation
4. Ensure CI passes
5. **Check PR status immediately after pushing**
6. Request review before merging

---

*This document serves as the primary context for AI assistants. It should be kept up-to-date as the project evolves.*