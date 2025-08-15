# Quick Start Guide

Get up and running with EOL Framework in 5 minutes!

## Prerequisites

Before starting, ensure you have:
- Python 3.11 or higher installed
- Redis Stack running (see [Installation](installation.md))

## Step 1: Install EOL

```bash
# Using uv (recommended - ultra fast)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install eol-rag-context

# Or using pip
pip install eol-rag-context
```

## Step 2: Start Redis

```bash
# Using Docker (recommended)
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest

# Verify Redis is running
redis-cli ping
# Should return: PONG
```

## Step 3: Your First RAG Pipeline

Create a file called `quickstart.py`:

```python
import asyncio
from eol.rag_context import DocumentIndexer, RedisStore, EmbeddingManager

async def main():
    # Initialize components
    redis_store = RedisStore()
    await redis_store.connect_async()
    
    embedding_manager = EmbeddingManager()
    indexer = DocumentIndexer(redis_store, embedding_manager)
    
    # Create sample documents
    documents = [
        {
            "id": "doc1",
            "content": "EOL is a RAG framework for building AI applications.",
            "metadata": {"category": "overview"}
        },
        {
            "id": "doc2", 
            "content": "RAG combines retrieval with generation for better AI responses.",
            "metadata": {"category": "concepts"}
        },
        {
            "id": "doc3",
            "content": "EOL uses Redis for high-performance vector search.",
            "metadata": {"category": "features"}
        }
    ]
    
    # Index documents
    print("Indexing documents...")
    for doc in documents:
        await indexer.index_document(doc)
    
    # Search for relevant content
    query = "What is EOL framework?"
    print(f"\nSearching for: {query}")
    
    results = await redis_store.search_similar(query, k=2)
    
    # Display results
    print("\nSearch Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.3f}")
        print(f"   Content: {result.content}")
        print(f"   Category: {result.metadata.get('category', 'N/A')}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 4: Run the Example

```bash
python quickstart.py
```

Expected output:
```
Indexing documents...

Searching for: What is EOL framework?

Search Results:
1. Score: 0.921
   Content: EOL is a RAG framework for building AI applications.
   Category: overview

2. Score: 0.756
   Content: EOL uses Redis for high-performance vector search.
   Category: features
```

## Step 5: Build Your Application

Now you can build on this foundation:

### Index Your Documents

```python
# Index a folder of documents
await indexer.index_folder("./my-documents")

# Index with custom chunking
from eol.rag_context import DocumentProcessor

processor = DocumentProcessor(
    chunk_size=500,
    chunk_overlap=50,
    chunk_strategy="semantic"
)

chunks = processor.process_file("large-document.pdf")
for chunk in chunks:
    await indexer.index_document(chunk)
```

### Advanced Search

```python
# Search with filters
results = await redis_store.search_similar(
    query="installation guide",
    k=5,
    filters={"category": "documentation"}
)

# Hybrid search (vector + keyword)
results = await redis_store.hybrid_search(
    query="Python async programming",
    k=10,
    keyword_weight=0.3,
    vector_weight=0.7
)
```

### Enable Semantic Caching

```python
from eol.rag_context import SemanticCache

# Initialize cache
cache = SemanticCache(redis_store, embedding_manager)

# Check cache before expensive operations
cached_response = await cache.get("How to install EOL?")

if not cached_response:
    # Perform expensive operation (e.g., LLM call)
    response = generate_response(query)
    
    # Cache the response
    await cache.set(query, response)
```

## Next Steps

Congratulations! You've built your first RAG pipeline with EOL. Here's what to explore next:

### Learn More
- [Configuration Guide](configuration.md) - Customize EOL for your needs
- [User Guide](../packages/eol-rag-context/user-guide/index.md) - Deep dive into features
- [API Reference](../packages/eol-rag-context/api-reference/index.md) - Complete API documentation

### Advanced Features
- [Knowledge Graphs](../packages/eol-rag-context/user-guide/advanced-features.md#knowledge-graphs) - Build entity relationships
- [File Watching](../packages/eol-rag-context/user-guide/advanced-features.md#file-watching) - Real-time document updates
- [MCP Server](../packages/eol-rag-context/user-guide/integrations.md#mcp-server) - Integrate with MCP clients

### Examples
- [Chat Application](../packages/eol-rag-context/examples/advanced-usage.md#chat-application)
- [Document Q&A](../packages/eol-rag-context/examples/basic-usage.md#document-qa)
- [Code Assistant](../packages/eol-rag-context/examples/advanced-usage.md#code-assistant)

## Common Issues

### Redis Connection Error
```bash
# Ensure Redis is running
docker ps | grep redis-stack

# If not running, start it
docker start redis-stack
```

### Import Error
```bash
# Ensure EOL is installed in your current environment
python -c "import eol.rag_context; print(eol.rag_context.__version__)"
```

### Performance Issues
- Reduce `chunk_size` for better granularity
- Increase `batch_size` for faster indexing
- Use `async` operations for better concurrency

## Get Help

- 📖 [Documentation](../index.md)
- 💬 [GitHub Discussions](https://github.com/eoln/eol/discussions)
- 🐛 [Report Issues](https://github.com/eoln/eol/issues)

---

Ready to build something amazing? Dive into the [User Guide](../packages/eol-rag-context/user-guide/index.md) for more!