# EOL RAG Context Tutorial

A step-by-step guide to using the EOL RAG Context MCP server for intelligent context management in your AI applications.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Basic Usage](#basic-usage)
4. [Indexing Documents](#indexing-documents)
5. [Searching & Retrieval](#searching--retrieval)
6. [Advanced Features](#advanced-features)
7. [Integration Examples](#integration-examples)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Introduction

The EOL RAG Context MCP server replaces static documentation with dynamic, intelligent context retrieval. It uses Redis 8's vector capabilities to provide:

- **Semantic Search**: Find relevant context based on meaning, not just keywords
- **Hierarchical Organization**: Concepts → Sections → Chunks
- **Real-time Updates**: Automatically index changes to your codebase
- **Smart Caching**: Optimize retrieval with 31% hit rate target

## Installation & Setup

### Prerequisites

- Python 3.11+
- Redis Stack 8.0+ (or Docker)
- 4GB+ RAM recommended

### Step 1: Install the Package

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install eol-rag-context

# Or install from source
git clone https://github.com/eoln/eol.git
cd eol/packages/eol-rag-context
pip install -e .
```

### Step 2: Start Redis

**Option A: Using Docker (Easiest)**
```bash
docker run -d \
  --name redis-rag \
  -p 6379:6379 \
  -p 8001:8001 \
  redis/redis-stack:latest
```

**Option B: Native Installation**
```bash
# macOS
brew install redis-stack

# Ubuntu/Debian
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update
sudo apt-get install redis-stack-server

# Start Redis
redis-stack-server
```

### Step 3: Create Configuration

Create `config.yaml`:

```yaml
redis:
  host: localhost
  port: 6379
  db: 0
  max_connections: 10

embedding:
  provider: sentence_transformers
  model_name: all-MiniLM-L6-v2
  dimension: 384
  batch_size: 32

indexing:
  chunk_size: 1000
  chunk_overlap: 200
  hierarchy_levels: 3
  
semantic_cache:
  enabled: true
  similarity_threshold: 0.9
  max_cache_size: 1000
  ttl_seconds: 3600
  
file_watcher:
  enabled: true
  watch_interval: 5
  debounce_seconds: 2
  
knowledge_graph:
  enabled: true
  max_depth: 3
  entity_types:
    - class
    - function
    - module
    - variable
```

## Basic Usage

### Starting the Server

```bash
# With default configuration
eol-rag-context

# With custom configuration
eol-rag-context config.yaml

# With environment variables
export REDIS_HOST=localhost
export REDIS_PORT=6379
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=your-key-here
eol-rag-context
```

### Using with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "eol-rag-context": {
      "command": "eol-rag-context",
      "args": ["config.yaml"],
      "env": {
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379"
      }
    }
  }
}
```

## Indexing Documents

### Index a Single File

```python
from eol.rag_context import EOLRAGContextServer

# Initialize server
server = EOLRAGContextServer()
await server.initialize()

# Index a single file
result = await server.index_directory("/path/to/file.py")
print(f"Indexed {result['chunks']} chunks from {result['file']}")
```

### Index a Directory

```python
# Index entire directory
result = await server.index_directory(
    "/path/to/project",
    recursive=True,
    patterns=["*.py", "*.md", "*.yaml"],
    ignore=["__pycache__", ".git", "node_modules"]
)

print(f"Indexed {result['indexed_files']} files")
print(f"Total chunks: {result['total_chunks']}")
print(f"Source ID: {result['source_id']}")
```

### Watch for Changes

```python
# Start watching a directory for changes
watch_result = await server.watch_directory(
    "/path/to/project",
    patterns=["*.py", "*.md"],
    auto_index=True
)

print(f"Watching with ID: {watch_result['watch_id']}")

# Stop watching
await server.unwatch_directory(watch_result['watch_id'])
```

## Searching & Retrieval

### Basic Search

```python
# Search for relevant context
results = await server.search_context(
    "How to implement authentication?",
    limit=5
)

for result in results:
    print(f"Score: {result['score']:.2f}")
    print(f"Content: {result['content'][:200]}...")
    print(f"Source: {result['metadata']['source']}")
    print("---")
```

### Hierarchical Search

```python
# Search at different hierarchy levels
results = await server.search_context(
    "database connection",
    hierarchy_level=1,  # Concepts only
    limit=3
)

# Get more detailed sections
for concept in results:
    sections = await server.search_context(
        "database connection",
        hierarchy_level=2,
        parent_id=concept['id'],
        limit=5
    )
```

### Using Filters

```python
# Search with metadata filters
results = await server.search_context(
    "error handling",
    filters={
        "file_type": "python",
        "module": "auth",
        "last_modified": {"$gte": "2024-01-01"}
    },
    limit=10
)
```

## Advanced Features

### Knowledge Graph Queries

```python
# Query the knowledge graph
graph = await server.query_knowledge_graph(
    entity="UserAuthentication",
    max_depth=2
)

print(f"Found {len(graph['entities'])} entities")
print(f"Found {len(graph['relationships'])} relationships")

# Visualize relationships
for rel in graph['relationships']:
    print(f"{rel['source']} --{rel['type']}--> {rel['target']}")
```

### Semantic Caching

```python
# Enable semantic caching for repeated queries
await server.optimize_context(target_hit_rate=0.31)

# First query (cache miss)
start = time.time()
results1 = await server.search_context("user authentication flow")
print(f"First query: {time.time() - start:.2f}s")

# Similar query (cache hit)
start = time.time()
results2 = await server.search_context("authentication process for users")
print(f"Cached query: {time.time() - start:.2f}s")

# Get cache statistics
stats = await server.get_stats()
print(f"Cache hit rate: {stats['cache']['hit_rate']:.2%}")
```

### Context Windows Management

```python
# Get optimized context for LLM
context = await server.get_context(
    "context://implement payment processing",
    max_tokens=4000,
    include_hierarchy=True
)

# Format for LLM consumption
formatted = "\n\n".join([
    f"## {doc['metadata'].get('header', 'Context')}\n{doc['content']}"
    for doc in context
])
```

## Integration Examples

### Example 1: Code Assistant

```python
import asyncio
from eol.rag_context import EOLRAGContextServer

async def code_assistant():
    # Initialize RAG server
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Index your codebase
    await server.index_directory(
        "/path/to/project",
        patterns=["*.py", "*.js", "*.md"]
    )
    
    # User query
    query = "How do I add a new API endpoint?"
    
    # Get relevant context
    context = await server.search_context(query, limit=5)
    
    # Build prompt for LLM
    prompt = "Based on the following context, answer the question.\n\n"
    prompt += "Context:\n"
    for ctx in context:
        prompt += f"- {ctx['content'][:500]}...\n"
    prompt += f"\nQuestion: {query}\nAnswer:"
    
    # Send to your LLM (Claude, GPT, etc.)
    # response = await llm.generate(prompt)
    
    return context

# Run the assistant
asyncio.run(code_assistant())
```

### Example 2: Documentation Search

```python
async def search_docs(query: str):
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Search only markdown files
    results = await server.search_context(
        query,
        filters={"file_type": "markdown"},
        hierarchy_level=2,  # Section level
        limit=10
    )
    
    # Group by document
    docs = {}
    for result in results:
        source = result['metadata']['source']
        if source not in docs:
            docs[source] = []
        docs[source].append(result)
    
    # Display results
    for doc, sections in docs.items():
        print(f"\n📄 {doc}")
        for section in sections:
            print(f"  - {section['metadata'].get('header', 'Section')}")
            print(f"    {section['content'][:100]}...")
```

### Example 3: Real-time Context Updates

```python
async def live_context_system():
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Set up file watching
    watch_id = await server.watch_directory(
        "/path/to/active/project",
        patterns=["*.py", "*.md"],
        recursive=True
    )
    
    print(f"Watching for changes... (ID: {watch_id})")
    
    # Simulate ongoing work
    while True:
        # Get latest context for current file
        current_file = "/path/to/active/project/main.py"
        
        context = await server.search_context(
            f"functions in {current_file}",
            filters={"source": current_file},
            limit=10
        )
        
        print(f"Found {len(context)} relevant pieces")
        
        # Wait before next check
        await asyncio.sleep(5)
```

### Example 4: CLI Tool

```python
#!/usr/bin/env python
"""rag-search - Command-line RAG search tool"""

import asyncio
import sys
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()

@app.command()
def search(
    query: str,
    path: Path = Path.cwd(),
    limit: int = 5,
    watch: bool = False
):
    """Search for context in your codebase."""
    
    async def run_search():
        from eol.rag_context import EOLRAGContextServer
        
        server = EOLRAGContextServer()
        await server.initialize()
        
        # Index if needed
        console.print(f"[blue]Indexing {path}...[/blue]")
        result = await server.index_directory(str(path))
        console.print(f"[green]Indexed {result['indexed_files']} files[/green]")
        
        # Search
        console.print(f"\n[yellow]Searching for: {query}[/yellow]\n")
        results = await server.search_context(query, limit=limit)
        
        # Display results
        table = Table(title="Search Results")
        table.add_column("Score", style="cyan")
        table.add_column("Source", style="magenta")
        table.add_column("Content", style="white")
        
        for r in results:
            table.add_row(
                f"{r['score']:.2f}",
                Path(r['metadata']['source']).name,
                r['content'][:100] + "..."
            )
        
        console.print(table)
        
        if watch:
            console.print("\n[blue]Watching for changes... Press Ctrl+C to stop[/blue]")
            await server.watch_directory(str(path))
            await asyncio.sleep(float('inf'))
    
    asyncio.run(run_search())

if __name__ == "__main__":
    app()
```

## Best Practices

### 1. Optimal Chunk Sizes

```yaml
# For code files
code_chunking:
  max_chunk_size: 500  # Smaller for precise context
  chunk_overlap: 100
  use_ast: true  # Parse code structure

# For documentation
doc_chunking:
  max_chunk_size: 1000  # Larger for narrative flow
  chunk_overlap: 200
  split_by_headers: true

# For structured data
structured_chunking:
  max_chunk_size: 300
  preserve_structure: true
```

### 2. Embedding Model Selection

| Use Case | Model | Dimension | Speed | Quality |
|----------|-------|-----------|-------|---------|
| Development | all-MiniLM-L6-v2 | 384 | Fast | Good |
| Production | all-mpnet-base-v2 | 768 | Medium | Better |
| High Accuracy | text-embedding-3-large | 3072 | Slow | Best |

### 3. Indexing Strategy

```python
# Index in priority order
async def smart_indexing(server, project_path):
    # 1. Index critical documentation first
    await server.index_directory(
        f"{project_path}/docs",
        priority=1
    )
    
    # 2. Index main source code
    await server.index_directory(
        f"{project_path}/src",
        priority=2,
        patterns=["*.py", "*.js"]
    )
    
    # 3. Index tests and examples
    await server.index_directory(
        f"{project_path}/tests",
        priority=3
    )
    
    # 4. Watch for changes
    await server.watch_directory(
        project_path,
        auto_index=True
    )
```

### 4. Performance Optimization

```python
# Batch operations for better performance
async def batch_index(server, files):
    batch_size = 10
    
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        tasks = [
            server.index_directory(f)
            for f in batch
        ]
        results = await asyncio.gather(*tasks)
        print(f"Indexed batch {i//batch_size + 1}")
```

### 5. Context Window Management

```python
def optimize_context_for_llm(contexts, max_tokens=4000):
    """Optimize context to fit in LLM window."""
    
    # Sort by relevance
    contexts.sort(key=lambda x: x['score'], reverse=True)
    
    # Track token count (rough estimate)
    result = []
    token_count = 0
    
    for ctx in contexts:
        # Estimate tokens (roughly 4 chars = 1 token)
        ctx_tokens = len(ctx['content']) // 4
        
        if token_count + ctx_tokens > max_tokens:
            # Truncate if needed
            remaining = max_tokens - token_count
            if remaining > 100:  # Worth including partial
                truncated = ctx['content'][:remaining * 4]
                ctx['content'] = truncated
                result.append(ctx)
            break
        
        result.append(ctx)
        token_count += ctx_tokens
    
    return result
```

## Troubleshooting

### Common Issues

#### 1. Redis Connection Failed

```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG

# Check Redis version (need 8.0+)
redis-server --version

# Check if RediSearch module is loaded
redis-cli MODULE LIST
```

#### 2. Slow Indexing

```python
# Use batch processing
await server.index_directory(
    path,
    batch_size=50,  # Increase batch size
    parallel_workers=4  # Use multiple workers
)

# Disable file watching during bulk indexing
await server.set_config({"file_watcher": {"enabled": False}})
await server.index_directory(path)
await server.set_config({"file_watcher": {"enabled": True}})
```

#### 3. Memory Issues

```yaml
# Adjust Redis memory settings
redis:
  max_memory: 2gb
  max_memory_policy: allkeys-lru

# Reduce embedding dimensions
embedding:
  model_name: all-MiniLM-L6-v2  # 384 dims vs 768
  
# Limit cache size
semantic_cache:
  max_cache_size: 500  # Reduce from 1000
```

#### 4. Poor Search Results

```python
# Tune similarity threshold
await server.optimize_context(
    similarity_threshold=0.85,  # Lower = more results
    rerank=True  # Enable result reranking
)

# Use query expansion
expanded_query = f"{query} {' '.join(synonyms)}"
results = await server.search_context(expanded_query)
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
eol-rag-context

# Or in Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Check

```python
async def health_check():
    server = EOLRAGContextServer()
    
    try:
        await server.initialize()
        stats = await server.get_stats()
        
        print("✅ Server Status: Healthy")
        print(f"📊 Documents: {stats['indexer']['total_documents']}")
        print(f"📦 Chunks: {stats['indexer']['total_chunks']}")
        print(f"💾 Cache Hit Rate: {stats['cache']['hit_rate']:.1%}")
        print(f"🔗 Graph Nodes: {stats['graph']['nodes']}")
        
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")
        return False
    
    return True

asyncio.run(health_check())
```

## Next Steps

1. **Explore Advanced Features**
   - Knowledge graph visualization
   - Custom embedding models
   - Fine-tuning for your domain

2. **Optimize for Production**
   - Set up monitoring
   - Configure alerts
   - Implement backup strategies

3. **Integrate with Your Tools**
   - IDE plugins
   - CI/CD pipelines
   - Chat interfaces

4. **Join the Community**
   - GitHub: https://github.com/eoln/eol
   - Issues: Report bugs and request features
   - Discussions: Share use cases and tips

## Conclusion

The EOL RAG Context MCP server provides powerful context management for AI applications. By following this tutorial, you can:

- ✅ Index any codebase or documentation
- ✅ Perform semantic search with high accuracy
- ✅ Keep context automatically updated
- ✅ Optimize retrieval with intelligent caching
- ✅ Build knowledge graphs of your code

Start with basic indexing and search, then gradually explore advanced features as your needs grow. The system is designed to scale from small projects to large codebases.

Happy coding! 🚀