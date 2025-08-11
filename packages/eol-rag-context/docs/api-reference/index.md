# API Reference

Complete API documentation for EOL RAG Context, automatically generated from docstrings.

## Overview

The EOL RAG Context API is organized into several key modules:

- **[Server](server.md)** - Main MCP server and entry point
- **[Configuration](config.md)** - Configuration management and validation
- **[Indexer](indexer.md)** - Document indexing and hierarchy creation
- **[Embeddings](embeddings.md)** - Embedding generation and management
- **[Redis Client](redis-client.md)** - Vector storage and retrieval
- **[Semantic Cache](semantic-cache.md)** - Intelligent caching layer
- **[Knowledge Graph](knowledge-graph.md)** - Entity and relationship extraction
- **[File Watcher](file-watcher.md)** - Real-time file monitoring
- **[Document Processor](document-processor.md)** - Multi-format document processing

## Quick Reference

### Core Classes

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `EOLRAGContextServer` | Main MCP server | `initialize()`, `index_directory()`, `search_context()` |
| `DocumentIndexer` | Document indexing | `index_file()`, `index_folder()` |
| `RedisVectorStore` | Vector storage | `store_document()`, `vector_search()` |
| `EmbeddingManager` | Embedding generation | `get_embedding()`, `batch_embed()` |

### Data Models

| Model | Purpose | Usage |
|-------|---------|-------|
| `SearchContextRequest` | Search parameters | MCP search tool requests |
| `QueryKnowledgeGraphRequest` | Graph query parameters | Knowledge graph queries |
| `RAGConfig` | System configuration | Server initialization |
| `IndexResult` | Indexing outcomes | Return from indexing operations |

## Type System

All APIs use comprehensive type hints compatible with Python 3.11+. Key types include:

```python
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from pydantic import BaseModel

# Common type aliases
DocumentData = Dict[str, Any]
EmbeddingVector = List[float]
SearchResults = List[Tuple[str, float, Dict[str, Any]]]
```

## Error Handling

All modules define custom exceptions with clear error messages:

```python
# Common exception patterns
try:
    result = await server.index_directory("/path/to/docs")
except FileNotFoundError:
    # Handle missing directory
except DocumentProcessingError as e:
    # Handle processing failures
except RedisConnectionError as e:
    # Handle Redis connectivity issues
```

## Navigation

Browse the API documentation by module, or use the search functionality to find specific methods, classes, or concepts.

Each API page includes:
- **Class/Function signatures** with full type annotations
- **Comprehensive descriptions** from docstrings
- **Parameter documentation** with types and defaults
- **Return value specifications** with type information
- **Usage examples** with practical code samples
- **Exception information** for error handling