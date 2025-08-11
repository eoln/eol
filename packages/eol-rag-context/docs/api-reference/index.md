# API Reference

Comprehensive API documentation for EOL RAG Context, automatically generated from source code docstrings.

## Overview

The EOL RAG Context API is organized into several core modules, each responsible for specific functionality within the intelligent context management system.

### Core Modules

#### 🎯 [Server](server.md)
The main MCP server implementation that orchestrates all components and provides the primary API interface.
- `EOLRAGContextServer` - Main server class
- MCP protocol implementation
- Request/response models
- Tool registration and management

#### ⚙️ [Configuration](config.md)
Configuration management system with validation and environment variable support.
- `RAGConfig` - Main configuration class
- Configuration validation
- Environment variable integration
- Default settings management

#### 🗄️ [Redis Client](redis-client.md)
Redis Stack integration for vector storage and retrieval.
- `RedisVectorStore` - Vector database operations
- Document storage and retrieval
- Vector similarity search
- Hierarchical search capabilities

#### 📄 [Document Processor](document-processor.md)
Intelligent document processing with format-specific handlers.
- `DocumentProcessor` - Main processing class
- Format-specific processors (Markdown, PDF, Code, JSON)
- Chunking strategies
- Metadata extraction

#### 📚 [Indexer](indexer.md)
Document indexing engine with hierarchical organization.
- `DocumentIndexer` - Indexing orchestration
- File and directory indexing
- Batch processing
- Index statistics and management

#### 🧠 [Embeddings](embeddings.md)
Embedding generation and management with multiple provider support.
- `EmbeddingManager` - Provider abstraction
- Sentence Transformers integration
- OpenAI embeddings support
- Batch processing and caching

#### 💾 [Semantic Cache](semantic-cache.md)
Intelligent caching system with semantic similarity matching.
- `SemanticCache` - Cache implementation
- Similarity-based retrieval
- Cache optimization (31% hit rate target)
- Performance metrics

#### 🕸️ [Knowledge Graph](knowledge-graph.md)
Automatic knowledge graph construction and querying.
- `KnowledgeGraphBuilder` - Graph construction
- Entity extraction
- Relationship discovery
- Graph querying and visualization

#### 👁️ [File Watcher](file-watcher.md)
Real-time file monitoring with intelligent reindexing.
- `FileWatcher` - File system monitoring
- Change detection and debouncing
- Selective reindexing
- Performance optimization

## Usage Patterns

### Basic Initialization

```python
from eol.rag_context import EOLRAGContextServer

# Initialize with default configuration
server = EOLRAGContextServer()
await server.initialize()

# Initialize with custom configuration
config = {
    "redis": {
        "url": "redis://localhost:6379"
    },
    "embedding": {
        "provider": "sentence_transformers",
        "model": "all-MiniLM-L6-v2"
    }
}
server = EOLRAGContextServer(config=config)
await server.initialize()
```

### Common Operations

#### Indexing Documents

```python
# Index a single file
result = await server.index_file(
    file_path="/path/to/document.md",
    force_reindex=True
)

# Index a directory
result = await server.index_directory(
    directory_path="/path/to/docs",
    recursive=True,
    file_patterns=["*.py", "*.md"],
    exclude_patterns=["*.pyc", "__pycache__/*"]
)
```

#### Searching Content

```python
# Basic search
results = await server.search_context({
    'query': 'user authentication',
    'max_results': 5,
    'similarity_threshold': 0.7
}, None)

# Advanced search with filters
results = await server.search_context({
    'query': 'database configuration',
    'max_results': 10,
    'filters': {
        'file_types': ['.yaml', '.json'],
        'date_range': {
            'after': '2024-01-01'
        }
    },
    'search_level': 'section'
}, None)
```

#### Knowledge Graph Operations

```python
# Query knowledge graph
from eol.rag_context.server import QueryKnowledgeGraphRequest

request = QueryKnowledgeGraphRequest(
    query="authentication system components",
    max_depth=2,
    include_relationships=True,
    entity_types=["class", "function"]
)

graph_results = await server.query_knowledge_graph(request, None)
```

## Type Definitions

The API uses comprehensive type hints throughout. Key types include:

- `SearchContextRequest` - Search request parameters
- `QueryKnowledgeGraphRequest` - Knowledge graph query parameters
- `IndexDirectoryRequest` - Directory indexing parameters
- `Document` - Document representation
- `Chunk` - Document chunk with metadata
- `SearchResult` - Search result with relevance scoring

## Error Handling

All API methods include comprehensive error handling:

```python
try:
    result = await server.index_directory("/path/to/docs")
except FileNotFoundError as e:
    print(f"Directory not found: {e}")
except PermissionError as e:
    print(f"Permission denied: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

### Batch Operations

For optimal performance when processing multiple files:

```python
# Use batch indexing for large collections
result = await server.index_directory(
    directory_path="/large/collection",
    batch_size=100,
    parallel_workers=4
)
```

### Caching

The semantic cache automatically optimizes performance:

```python
# Configure caching for optimal performance
config = {
    "caching": {
        "enabled": True,
        "target_hit_rate": 0.31,
        "ttl_seconds": 3600
    }
}
```

### Memory Management

For large-scale operations:

```python
# Configure memory-efficient processing
config = {
    "indexing": {
        "chunk_buffer_size": 50,
        "memory_limit_mb": 2048,
        "gc_interval": 100
    }
}
```

## Extension Points

The API is designed for extensibility:

### Custom Embedding Providers

```python
from eol.rag_context.embeddings import EmbeddingProvider

class CustomEmbedding(EmbeddingProvider):
    async def get_embedding(self, text: str) -> np.ndarray:
        # Custom implementation
        pass

server.register_embedding_provider('custom', CustomEmbedding)
```

### Custom Document Processors

```python
from eol.rag_context.document_processor import DocumentProcessor

class CustomProcessor(DocumentProcessor):
    def process_custom_format(self, content: str) -> List[Chunk]:
        # Custom processing logic
        pass
```

## Best Practices

1. **Always use async/await** - All API methods are asynchronous
2. **Handle errors gracefully** - Implement proper error handling
3. **Configure for your use case** - Tune settings for optimal performance
4. **Monitor performance** - Use built-in metrics and monitoring
5. **Clean up resources** - Always close connections when done

```python
try:
    server = EOLRAGContextServer()
    await server.initialize()
    # ... perform operations
finally:
    await server.close()
```

## Module Documentation

Detailed documentation for each module is available in the following sections:

- [Server Module](server.md) - Core server implementation
- [Configuration Module](config.md) - Configuration management
- [Redis Client Module](redis-client.md) - Vector database operations
- [Document Processor Module](document-processor.md) - Document processing
- [Indexer Module](indexer.md) - Indexing engine
- [Embeddings Module](embeddings.md) - Embedding management
- [Semantic Cache Module](semantic-cache.md) - Caching system
- [Knowledge Graph Module](knowledge-graph.md) - Graph operations
- [File Watcher Module](file-watcher.md) - File monitoring

## Version Information

- **Current Version**: 1.0.0
- **Python Support**: 3.11+
- **Redis Stack**: 7.2+
- **API Stability**: Stable

## Need Help?

- 📖 Check the [User Guide](../user-guide/) for tutorials and guides
- 💡 See [Examples](../examples/) for practical code samples
- 🐛 Report issues on [GitHub](https://github.com/eoln/eol)
- 💬 Join community discussions for support