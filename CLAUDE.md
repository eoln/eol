# EOL AI Framework - Project Context for AI Assistants

## Overview
EOL is an AI Framework for building modern LLM applications. The framework provides tools and services for intelligent context management, with the `eol-rag-context` MCP server replacing static documentation with dynamic, Redis 8-backed RAG systems. This document provides essential context and guidelines for AI assistants working on this project.

## Key Components

### eol-rag-context MCP Server
The core context management service that:
- Indexes project files hierarchically (concepts → sections → chunks)
- Provides intelligent context retrieval via RAG
- Optimizes context structure based on 2024 LLM best practices
- Eliminates need for static `.claude/context` files
- Serves context dynamically via MCP protocol

## Core Principles

### 1. Two-Phase Development Model
- **Phase 1: Prototyping** - Use natural language specifications in `.eol.md` files
- **Phase 2: Implementation** - Convert to deterministic Python code
- **Hybrid Mode** - Mix both phases, switch ad-hoc based on needs
- Always consider which phase is appropriate for the current task

### 2. Context-First Development
- Leverage Redis v8 vector database for context storage
- Implement semantic caching (target: 31% hit rate)
- Use content-aware chunking (AST for code, semantic for text)
- Manage LLM context window efficiently

### 3. Dependency-Driven Architecture
- Features can depend on other features
- Support 6 dependency types: features, MCP servers, services, packages, containers, models
- Always resolve dependencies before execution
- Implement fallback mechanisms for critical dependencies

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

## Project Structure

```
eol/
├── packages/          # Monorepo packages (uv workspace)
│   ├── eol-core/     # Core engine (parser, phase manager, context)
│   ├── eol-cli/      # CLI interface
│   ├── eol-mcp/      # MCP server implementation
│   └── eol-redis/    # Redis integration
├── examples/         # Example .eol.md files
├── features/         # Feature specifications
└── .claude/context/  # RAG and technical documentation
```

## Code Style and Conventions

### Python Code
- Use type hints for all function signatures
- Async/await for I/O operations
- Dataclasses for data structures
- Abstract base classes for interfaces
- Comprehensive docstrings with examples

### File Naming
- Feature files: `<name>.eol.md`
- Test files: `<name>.test.eol.md`
- Python modules: lowercase with underscores
- Classes: PascalCase
- Functions/variables: snake_case

### Error Handling
- Custom exception classes for each error type
- Always provide context in error messages
- Implement retry logic with exponential backoff
- Use circuit breakers for external services

## Implementation Guidelines

### When Adding New Features

1. **Start with Documentation**
   - Create or update `.eol.md` specification
   - Write test cases in `.test.eol.md`
   - Document dependencies explicitly

2. **Follow the Phase Pattern**
   ```python
   # Always support both phases
   async def execute_operation(phase: ExecutionPhase):
       if phase == ExecutionPhase.PROTOTYPING:
           return await execute_via_llm()
       elif phase == ExecutionPhase.IMPLEMENTATION:
           return await execute_deterministic()
       else:  # HYBRID
           # Try implementation first, fallback to prototype
   ```

3. **Manage Dependencies**
   ```python
   # Always resolve dependencies first
   resolver = DependencyResolver(project_root)
   deps = await resolver.resolve_feature(feature_path, phase)
   ```

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
- **Validate all inputs** - Especially in .eol.md files
- **Sanitize LLM outputs** - Before execution
- **Implement rate limiting** - For all external APIs
- **Use least privilege** - For all service accounts

## Performance Targets

- **Response Time**: < 2s for prototyping, < 500ms for implementation
- **Cache Hit Rate**: > 31% for semantic cache
- **Context Window Usage**: Keep below 80% normally
- **Dependency Resolution**: < 1s for typical features

## Common Patterns

### Dependency Injection
```python
class FeatureExecutor:
    def __init__(self, dependencies: Dict[str, Any]):
        self.deps = dependencies
    
    async def execute(self):
        model = self.deps.get('models:claude-3-opus')
        redis = self.deps.get('services:redis')
```

### Phase Switching
```python
# Allow runtime phase switching
phase_manager.switch_phase(
    feature="payment-processor",
    to_phase=ExecutionPhase.IMPLEMENTATION,
    operations=["process_payment"]
)
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

1. **Enable verbose logging**: `eol run feature.eol.md --verbose`
2. **Check dependency graph**: `eol deps graph feature.eol.md`
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
- ✅ Follow the two-phase model
- ✅ Optimize for context window
- ✅ Use semantic caching

### DON'T:
- ❌ Hardcode credentials
- ❌ Skip error handling
- ❌ Ignore phase requirements
- ❌ Mix prototyping and implementation code
- ❌ Exceed context window limits
- ❌ Use synchronous I/O
- ❌ Forget dependency resolution
- ❌ Skip tests

## Getting Help

- Check `.claude/context/` for detailed documentation
- Review `examples/` for reference implementations
- Use `eol --help` for CLI documentation
- Refer to test files for usage examples

## Contributing

When contributing to EOL:
1. Follow the code style guide
2. Write tests for new features
3. Update documentation
4. Ensure CI passes
5. Request review before merging

---

*This document serves as the primary context for AI assistants. It should be kept up-to-date as the project evolves.*