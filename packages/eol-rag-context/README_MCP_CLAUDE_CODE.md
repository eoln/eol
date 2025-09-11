# EOL RAG Context MCP Server for Claude Code CLI

## Overview

This MCP (Model Context Protocol) server provides intelligent RAG (Retrieval-Augmented Generation) capabilities to Claude Code CLI, enabling semantic search, code indexing, and context-aware assistance across all your projects.

## Features

- 🔍 **Semantic Search**: Find code and documentation using natural language queries
- 📁 **Automatic Indexing**: Index entire project directories with smart chunking
- 🧠 **Hierarchical Understanding**: Concepts → Sections → Chunks organization
- ⚡ **Performance**: Redis v8 vector database for fast retrieval
- 🔄 **Incremental Updates**: Only reindex changed files
- 📊 **Rich Metadata**: Track file types, languages, git info, and more

## Installation

### Prerequisites

1. **Python 3.13+**
2. **Redis 8.0+** with RediSearch module
3. **uv** package manager
4. **Claude Code CLI** installed

### Quick Setup

```bash
# 1. Navigate to the package directory
cd /Users/eoln/Devel/eol/packages/eol-rag-context

# 2. Run the setup script
./setup_mcp_claude_code.sh

# 3. Restart Claude Code CLI to load the MCP server
```

## Manual Setup

### 1. Install Dependencies

```bash
cd /Users/eoln/Devel/eol
uv sync
```

### 2. Start Redis

```bash
# Using the custom config with RediSearch
/usr/local/opt/redis/bin/redis-server /Users/eoln/Devel/eol/packages/eol-rag-context/redis-v8.conf
```

### 3. Configure Claude Code

Add to your Claude Code configuration:

```json
{
  "mcpServers": {
    "eol-rag-context": {
      "command": "/Users/eoln/Devel/eol/.venv/bin/python",
      "args": [
        "/Users/eoln/Devel/eol/packages/eol-rag-context/mcp_launcher_final.py"
      ],
      "env": {
        "PYTHONPATH": "/Users/eoln/Devel/eol/packages/eol-rag-context/src"
      }
    }
  }
}
```

## Usage in Claude Code CLI

### Basic Commands

```bash
# Index current project
claude "Index the current directory for semantic search"

# Search for specific functionality
claude "Find all authentication related code"

# Get indexing statistics
claude "Show RAG indexing stats"

# List indexed sources
claude "What files have been indexed?"

# Force reindex
claude "Reindex all files even if unchanged"
```

### Advanced Usage

```bash
# Search with context
claude "Find the database connection setup and explain how it works"

# Cross-file analysis
claude "Show me all API endpoints and their authentication methods"

# Find related code
claude "Find code similar to the user authentication module"

# Temporal queries (for XML event data)
claude "Find all events happening in June 2025"
```

## MCP Tools Available

### 1. `get_stats`

Returns current indexing statistics including document count, chunks, and sources.

### 2. `list_sources`

Lists all indexed sources with metadata like file count, chunks, and indexing time.

### 3. `index_directory`

Indexes a directory with options:

- `path`: Directory to index (required)
- `recursive`: Include subdirectories (default: true)
- `force_reindex`: Reindex unchanged files (default: false)

### 4. `search_context`

Searches indexed content:

- `query`: Search query (required)
- `max_results`: Maximum results to return (default: 10)

### 5. `test_sandbox`

Tests environment access and permissions.

## Project-Specific Setup

For each project, the MCP server will create:

- `.rag-data/`: Indexed data storage
- `.rag-index/`: Search index files

Add these to your `.gitignore`:

```
.rag-data/
.rag-index/
```

## Performance Optimization

### Indexing Performance

- **Target**: >10 documents/second
- **Batch Processing**: Processes multiple files concurrently
- **Smart Chunking**: Content-aware splitting for optimal context

### Search Performance

- **Latency**: <100ms for 10k documents
- **Vector Search**: HNSW algorithm for fast similarity search
- **Caching**: Semantic cache with 31% hit rate target

## Troubleshooting

### Redis Connection Issues

```bash
# Check if Redis is running
redis-cli ping

# Check Redis version
redis-cli INFO server | grep redis_version

# Restart Redis with custom config
redis-cli shutdown
/usr/local/opt/redis/bin/redis-server redis-v8.conf
```

### MCP Server Not Connecting

```bash
# Test MCP server directly
/Users/eoln/Devel/eol/.venv/bin/python mcp_launcher_final.py

# Check logs
tail -f ~/.eol-rag-context/mcp_server_*.log
```

### Indexing Issues

```bash
# Clear Redis database and start fresh
redis-cli FLUSHDB

# Force reindex with Claude
claude "Force reindex the entire project"
```

## Configuration

### Environment Variables

- `EOL_RAG_DATA_DIR`: Data storage directory (default: `.rag-data`)
- `EOL_RAG_INDEX_DIR`: Index storage directory (default: `.rag-index`)
- `PYTHONPATH`: Must include `/src` directory

### File Patterns

By default, indexes:

- Code: `*.py`, `*.js`, `*.ts`, `*.java`, `*.go`, `*.rs`, etc.
- Docs: `*.md`, `*.txt`, `*.rst`, `*.adoc`
- Config: `*.json`, `*.yaml`, `*.xml`, `*.toml`

## Development

### Running Tests

```bash
cd /Users/eoln/Devel/eol/packages/eol-rag-context
python -m pytest tests/
```

### Monitoring Performance

The MCP server logs performance metrics:

```
📊 PERFORMANCE | index_directory | Duration: 4.368s | Memory: 104.7MB | files: 714, chunks: 357
```

## Architecture

```
Claude Code CLI
      ↓
MCP Protocol (JSON-RPC)
      ↓
eol-rag-context MCP Server
      ↓
Document Processor → Embedding Manager → Redis Vector Store
      ↓                    ↓                     ↓
  Chunking           Vectorization         HNSW Index
```

## License

GPL-3.0

## Support

For issues or questions, check:

- Logs: `~/.eol-rag-context/mcp_server_*.log`
- Redis: `redis-cli MONITOR`
- Debug: Set `EOL_DEBUG=true` for verbose logging
