# Server API

The main MCP server implementation for EOL RAG Context.

## EOLRAGContextServer

::: eol.rag_context.server.EOLRAGContextServer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Request Models

### SearchContextRequest

::: eol.rag_context.server.SearchContextRequest
    options:
      show_root_heading: true
      heading_level: 4

### QueryKnowledgeGraphRequest  

::: eol.rag_context.server.QueryKnowledgeGraphRequest
    options:
      show_root_heading: true
      heading_level: 4

## Usage Examples

### Basic Server Setup

```python
from eol.rag_context import EOLRAGContextServer
from eol.rag_context.config import RAGConfig

# With default configuration
server = EOLRAGContextServer()
await server.initialize()

# With custom configuration
config = RAGConfig(
    redis_host="localhost",
    redis_port=6379,
    embedding_provider="sentence-transformers"
)
server = EOLRAGContextServer(config)
await server.initialize()
```

### Document Indexing

```python
# Index a single directory
result = await server.index_directory(
    "/path/to/documents",
    recursive=True,
    file_patterns=["*.py", "*.md"]
)
print(f"Indexed {result['indexed_files']} files")

# Index a single file  
result = await server.index_file("/path/to/document.py")
print(f"Created {result['total_chunks']} chunks")
```

### Searching for Context

```python
from eol.rag_context.server import SearchContextRequest

# Create search request
request = SearchContextRequest(
    query="How to implement authentication?",
    max_results=10,
    min_relevance=0.7,
    hierarchy_level=3  # Search at chunk level
)

# Perform search
results = await server.search_context(request, None)

# Process results
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Content: {result['content'][:200]}...")
    print(f"Source: {result['metadata']['source']}")
```

### Knowledge Graph Queries

```python
from eol.rag_context.server import QueryKnowledgeGraphRequest

# Query the knowledge graph
request = QueryKnowledgeGraphRequest(
    query="UserAuthentication", 
    max_depth=2,
    max_entities=20
)

graph = await server.query_knowledge_graph(request, None)
print(f"Found {len(graph['entities'])} entities")
print(f"Found {len(graph['relationships'])} relationships")
```

## MCP Integration

The server automatically registers MCP tools for integration with Claude Desktop and other MCP clients. Tools are available at:

- `index_directory` - Index documents in a directory
- `search_context` - Search for relevant context
- `query_knowledge_graph` - Explore entity relationships
- `watch_directory` - Monitor directory for changes

See the [MCP Integration Guide](../user-guide/integrations.md) for complete setup instructions.