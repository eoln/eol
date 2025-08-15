# EOL RAG Context Examples

This directory contains example scripts demonstrating how to use the EOL RAG Context MCP server.

## Prerequisites

Before running the examples, make sure you have:

1. **Redis Running**:

```bash
docker run -d -p 6379:6379 redis/redis-stack:latest
```

2. **Dependencies Installed**:

```bash
pip install -r ../requirements.txt
```

## Examples

### 1. Quick Start (`quick_start.py`)

A simple introduction to basic RAG operations.

```bash
python quick_start.py
```

**Features demonstrated:**

- Server initialization
- Directory indexing
- Context search
- Statistics retrieval

### 2. Code Assistant (`code_assistant.py`)

An interactive AI code assistant that can answer questions about your codebase.

```bash
# Analyze current directory
python code_assistant.py

# Analyze specific project
python code_assistant.py /path/to/project
```

**Features demonstrated:**

- Project indexing with filters
- Interactive Q&A
- Code implementation search
- Improvement suggestions
- Knowledge graph queries

**Commands:**

- `help` - Show available commands
- `stats` - Display indexing statistics
- `find <name>` - Find implementations
- `improve <code>` - Get improvement suggestions
- Ask any question about the codebase

### 3. Documentation Search (`doc_search.py`)

Search and retrieve documentation efficiently.

```bash
python doc_search.py "authentication" /path/to/docs
```

**Features demonstrated:**

- Markdown-specific indexing
- Hierarchical search
- Result grouping
- Snippet extraction

### 4. Real-time Monitor (`realtime_monitor.py`)

Monitor a directory and maintain up-to-date context.

```bash
python realtime_monitor.py /path/to/watch
```

**Features demonstrated:**

- File watching
- Automatic re-indexing
- Change detection
- Live updates

### 5. API Server (`api_server.py`)

REST API wrapper for the RAG context server.

```bash
python api_server.py
```

**Endpoints:**

- `POST /index` - Index a directory
- `GET /search?q=query` - Search for context
- `GET /stats` - Get statistics
- `POST /watch` - Start watching directory

### 6. Benchmark (`benchmark.py`)

Performance testing and optimization.

```bash
python benchmark.py /path/to/test
```

**Tests:**

- Indexing speed
- Search latency
- Cache performance
- Memory usage

## Common Patterns

### Initialize Server

```python
from eol.rag_context import EOLRAGContextServer

server = EOLRAGContextServer()
await server.initialize()
```

### Index Files

```python
result = await server.index_directory(
    "/path/to/project",
    patterns=["*.py", "*.md"],
    ignore=["__pycache__", ".git"]
)
```

### Search Context

```python
results = await server.search_context(
    "your query",
    limit=5
)

for result in results:
    print(f"Score: {result['score']}")
    print(f"Content: {result['content']}")
```

### Watch for Changes

```python
watch_id = await server.watch_directory(
    "/path/to/watch",
    auto_index=True
)
```

## Tips

1. **Start Small**: Begin with `quick_start.py` to understand basics
2. **Use Filters**: Improve search accuracy with file type filters
3. **Adjust Chunk Size**: Smaller chunks for code, larger for docs
4. **Enable Caching**: Improves performance for repeated queries
5. **Monitor Stats**: Use statistics to optimize configuration

## Troubleshooting

### Redis Connection Error

```bash
# Check if Redis is running
redis-cli ping

# If not, start Redis
docker run -d -p 6379:6379 redis/redis-stack:latest
```

### Import Error

```bash
# Install from parent directory
cd ..
pip install -e .
```

### Slow Performance

- Reduce chunk size for faster indexing
- Enable semantic caching
- Use batch operations
- Consider using a faster embedding model

## Advanced Usage

For more advanced usage patterns, see:

- [TUTORIAL.md](../TUTORIAL.md) - Complete tutorial
- [API Documentation](../docs/api.md) - API reference
- [Configuration Guide](../docs/configuration.md) - Configuration options

## Contributing

Feel free to add your own examples! Please follow the existing pattern:

1. Clear documentation in the script
2. Error handling
3. Help text for CLI scripts
4. README entry explaining the example
