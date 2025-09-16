# EOL RAG Context Examples

Example scripts demonstrating the EOL RAG Context MCP server with Claude Code.

## Available Examples

### 1. Quick Start (`quick_start.py`)

Basic introduction to RAG operations - indexing and searching documents.

```bash
# Run directly
uv run python examples/quick_start.py

# Or use with Claude Code
claude "Run the quick_start.py example and explain what it's doing"
```

### 2. Code Assistant (`code_assistant.py`)

Interactive AI assistant that answers questions about your codebase.

```bash
# Analyze current directory
uv run python examples/code_assistant.py

# Analyze specific project
uv run python examples/code_assistant.py /path/to/project

# Use with Claude Code
claude "Use the code assistant to analyze the src directory"
```

### 3. RAG CLI (`rag_cli.py`)

Command-line interface for RAG operations.

```bash
# Index files
uv run python examples/rag_cli.py index /path/to/docs

# Search
uv run python examples/rag_cli.py search "authentication"

# Get stats
uv run python examples/rag_cli.py stats
```

## Using with Claude Code

The best way to use these examples is through Claude Code:

```bash
# Ask Claude to run examples
claude "Run the quick_start.py example"

# Get help understanding the code
claude "Explain how the code_assistant.py example works"

# Modify examples for your needs
claude "Modify the rag_cli.py to index only Python files"
```

## Prerequisites

1. **Redis 8.2+** running:

```bash
docker run -d -p 6379:6379 redis:8.2-alpine
```

2. **Dependencies installed**:

```bash
uv sync
```

## Common Patterns

### Initialize the Server

```python
from eol.rag_context import EOLRAGContextServer

server = EOLRAGContextServer()
await server.initialize()
```

### Index Documents

```python
result = await server.start_indexing(
    "/path/to/project",
    file_patterns=["*.py", "*.md"]
)
```

### Search for Context

```python
results = await server.search_context(
    "your query here",
    top_k=5
)
```

## Tips

- Start with `quick_start.py` to understand basics
- Use Claude Code to explore and modify examples
- Enable verbose logging with `--verbose` flag
- Check Redis connection if you encounter errors

## Troubleshooting

### Redis Connection Error

```bash
# Check Redis is running
redis-cli ping

# If not, start Redis
docker run -d -p 6379:6379 redis:8.2-alpine
```

### Module Import Error

```bash
# Ensure you're in the right directory
cd eol/packages/eol-rag-context

# Install in development mode
uv sync
```
