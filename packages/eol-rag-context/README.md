# EOL RAG Context MCP Server

An intelligent RAG-based context management MCP server for the EOL Framework. Replaces static context files with dynamic, Redis 8-backed retrieval system.

## Features

- 🔍 **Hierarchical Indexing**: 3-level structure (concepts → sections → chunks)
- 🧠 **Knowledge Graph**: Automatic entity extraction and relationship discovery
- ⚡ **Real-time Updates**: File watcher with automatic reindexing
- 💾 **Semantic Caching**: 31% hit rate optimization with adaptive threshold
- 📁 **Multi-format Support**: Markdown, PDF, DOCX, JSON/YAML, source code
- 🎯 **Precise Localization**: Track exact line/char positions in source files
- 🔄 **MCP Protocol**: Full Model Context Protocol implementation

## Installation

```bash
pip install eol-rag-context
```

### Dependencies

- Redis Stack 8.0+ (with RediSearch module)
- Python 3.11+

## Testing

The project includes comprehensive unit and integration tests with 80% coverage target.

### Quick Test

```bash
# Run all tests automatically (starts/stops Redis)
./test_all.sh
```

### Manual Testing

```bash
# Start Redis
docker run -d -p 6379:6379 redis/redis-stack:latest

# Run tests with coverage
pytest tests/ --cov=eol.rag_context --cov-report=term

# Stop Redis
docker stop $(docker ps -q --filter ancestor=redis/redis-stack:latest)
```

### Coverage Status

- **Current**: 80% ✅
- **Unit Tests**: 43%
- **Integration Tests**: +37%

See `.claude/context/testing.md` for detailed testing documentation.

## Quick Start

### 1. Start Redis

```bash
# Using Docker (recommended)
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

# Or use local Redis Stack
redis-stack-server
```

### 2. Run the Server

```bash
# With default configuration
eol-rag-context

# With custom configuration
eol-rag-context config.yaml
```

### 3. Use with Claude Desktop

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "eol-rag-context": {
      "command": "eol-rag-context",
      "env": {
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379"
      }
    }
  }
}
```

## Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=secret
REDIS_DB=0

# Embedding Configuration
EMBEDDING_PROVIDER=sentence-transformers  # or openai
EMBEDDING_MODEL=all-MiniLM-L6-v2
OPENAI_API_KEY=sk-...  # If using OpenAI

# Cache Configuration
CACHE_ENABLED=true
CACHE_TTL=3600
CACHE_SIMILARITY_THRESHOLD=0.97

# Context Configuration
MAX_CONTEXT_TOKENS=32000
DEFAULT_TOP_K=10
MIN_RELEVANCE_SCORE=0.7
```

### Configuration File

```yaml
# config.yaml
redis:
  host: localhost
  port: 6379
  db: 0

embedding:
  provider: sentence-transformers
  model_name: all-mpnet-base-v2
  dimension: 768

cache:
  enabled: true
  ttl_seconds: 3600
  similarity_threshold: 0.97
  target_hit_rate: 0.31

context:
  max_context_tokens: 32000
  default_top_k: 10
  use_hierarchical_retrieval: true

document:
  file_patterns:
    - "*.md"
    - "*.py"
    - "*.js"
    - "*.ts"
    - "*.pdf"
    - "*.docx"
```

## MCP Tools

### Index Directory

Index a directory of documents with hierarchical structure:

```typescript
{
  "tool": "index_folder",
  "params": {
    "folder_path": "/path/to/docs",
    "recursive": true,
    "file_patterns": ["*.md", "*.py"],
    "source_id": "optional_source_id"
  }
}
```

### Search Context

Search for relevant context using vector similarity:

```typescript
{
  "tool": "vector_search",
  "params": {
    "query": "How does authentication work?",
    "hierarchy_level": 3,
    "k": 10,
    "min_score": 0.7
  }
}
```

### Query Knowledge Graph

Explore entity relationships:

```typescript
{
  "tool": "query_subgraph",
  "params": {
    "query": "authentication system",
    "max_depth": 2,
    "include_relationships": true
  }
}
```

### Watch Directory

Monitor a directory for changes:

```typescript
{
  "tool": "watch_folder",
  "params": {
    "folder_path": "/path/to/project",
    "recursive": true,
    "file_patterns": ["*.py", "*.md"],
    "debounce_seconds": 2
  }
}
```

## MCP Resources

### Get Context

Retrieve optimized context for a query:

```
GET context://search/{query}
```

### List Sources

List all indexed sources:

```
GET sources://list
```

### Get Statistics

Get server statistics:

```
GET stats://server
```

## Architecture

```
┌─────────────────────────────────────────┐
│           MCP Interface Layer            │
│  (Tools, Resources, Prompts)             │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│          Context Engine Layer            │
│  (Hierarchical Organization, Synthesis)  │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│         Document Processing Layer        │
│  (Multi-format, Chunking, Metadata)      │
└─────────────────────────────────────────┘
                    │
┌─────────────────────────────────────────┐
│           Redis 8 Storage Layer          │
│  (HNSW Vectors, Semantic Cache, KG)      │
└─────────────────────────────────────────┘
```

## Testing

### Run Tests

```bash
# Unit tests only
./run_tests.sh

# With Redis integration
./run_tests.sh --redis

# All tests
./run_tests.sh --all
```

### Docker Test Environment

```bash
# Start test Redis
docker-compose -f docker-compose.test.yml up -d

# Run tests
pytest --redis

# Stop test Redis
docker-compose -f docker-compose.test.yml down
```

See [TESTING.md](TESTING.md) for comprehensive testing documentation.

## Advanced Features

### Hierarchical Indexing

Documents are indexed at three levels:

1. **Concepts**: High-level abstractions
2. **Sections**: Grouped related content
3. **Chunks**: Fine-grained retrievable units

### Knowledge Graph

Automatically extracts:

- **Entities**: Functions, classes, topics, technologies
- **Relationships**: Dependencies, references, similarities
- **Patterns**: Common structures, hubs, communities

### Semantic Caching

- Targets 31% hit rate based on research
- Adaptive threshold adjustment
- Similarity-based retrieval
- LRU eviction policy

### File Watching

- Real-time monitoring with watchdog
- Debouncing for rapid changes
- Batch processing for efficiency
- Fallback polling mode

## Performance

- **Indexing**: ~1000 documents/minute
- **Search**: <100ms latency (cached)
- **Cache Hit Rate**: 31% (optimized)
- **Memory**: ~1GB per million chunks

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/eoln/eol.git
cd eol/packages/eol-rag-context

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

### Project Structure

```
eol-rag-context/
├── src/eol/rag_context/
│   ├── server.py           # MCP server implementation
│   ├── config.py           # Configuration management
│   ├── redis_client.py     # Redis vector operations
│   ├── document_processor.py # Multi-format processing
│   ├── indexer.py          # Document indexing
│   ├── embeddings.py       # Embedding management
│   ├── semantic_cache.py   # Caching layer
│   ├── knowledge_graph.py  # Graph construction
│   └── file_watcher.py     # Real-time monitoring
├── tests/
│   ├── test_*.py           # Test files
│   └── conftest.py         # Test fixtures
├── docker-compose.test.yml # Test environment
└── run_tests.sh            # Test runner
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

GPL-3.0 - See [LICENSE](../../LICENSE) file

## Support

- [GitHub Issues](https://github.com/eoln/eol/issues)
- [Documentation](https://github.com/eoln/eol/wiki)

## Acknowledgments

Built with:

- [FastMCP](https://github.com/fastmcp/fastmcp) - MCP framework
- [Redis Stack](https://redis.io/docs/stack/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- Research on optimal LLM context structures (2024)
