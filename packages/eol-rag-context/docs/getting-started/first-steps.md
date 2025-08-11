# First Steps

This tutorial walks you through your first RAG context indexing and retrieval session, from setup to search results.

## Prerequisites

Before starting, ensure you have:

- **Python 3.11+** installed
- **Redis Stack 8+** running (see [Installation](installation.md))
- **EOL RAG Context** installed: `pip install eol-rag-context`

## Quick Verification

Let's verify everything is working:

```bash
# Check EOL RAG Context is installed
eol-rag-context --version

# Verify Redis Stack is running
redis-cli ping
# Should return: PONG

# Test Redis vector search capability
redis-cli FT._LIST
# Should return list of indexes (may be empty initially)
```

## Step 1: Start the MCP Server

The EOL RAG Context runs as an MCP (Model Context Protocol) server that Claude Desktop can connect to.

```bash
# Start with default configuration
eol-rag-context serve

# Or with custom configuration
eol-rag-context serve --config config.yaml

# Or with environment variables
EOL_REDIS_URL="redis://localhost:6379" eol-rag-context serve
```

You should see output similar to:
```
INFO     Starting EOL RAG Context MCP Server
INFO     Redis connected: localhost:6379/0
INFO     Embedding provider: sentence_transformers (all-MiniLM-L6-v2)
INFO     Server listening on stdio
```

## Step 2: Create Sample Documents

Let's create some sample documents to index:

```bash
# Create a sample project directory
mkdir sample-project && cd sample-project

# Create sample Python file
cat > main.py << 'EOF'
"""
Main application module for the sample project.

This module demonstrates various Python features including
classes, functions, and error handling.
"""

import logging
from typing import List, Dict, Optional

class DataProcessor:
    """A sample data processing class."""
    
    def __init__(self, name: str):
        """Initialize the processor with a name."""
        self.name = name
        self.processed_count = 0
    
    def process_data(self, data: List[str]) -> Dict[str, int]:
        """Process a list of strings and count their lengths."""
        result = {}
        for item in data:
            result[item] = len(item)
            self.processed_count += 1
        return result

def main():
    """Main application entry point."""
    processor = DataProcessor("sample")
    sample_data = ["hello", "world", "python", "programming"]
    results = processor.process_data(sample_data)
    print(f"Processed {len(results)} items")

if __name__ == "__main__":
    main()
EOF

# Create sample Markdown documentation
cat > README.md << 'EOF'
# Sample Project

This is a sample project for demonstrating EOL RAG Context capabilities.

## Features

- **Data Processing**: Process lists of strings
- **Length Calculation**: Calculate string lengths
- **Result Storage**: Store results in dictionaries

## Quick Start

1. Import the DataProcessor class
2. Create an instance with a name
3. Call process_data with your data

```python
from main import DataProcessor

processor = DataProcessor("my_processor")
results = processor.process_data(["hello", "world"])
print(results)  # {'hello': 5, 'world': 5}
```

## Architecture

The project follows a simple architecture:

- **DataProcessor**: Main processing class
- **process_data()**: Core processing method
- **main()**: Entry point function

## Use Cases

This pattern is useful for:
- Text processing applications
- Data transformation pipelines
- Simple ETL operations
EOF

# Create configuration file
cat > config.json << 'EOF'
{
  "processor_settings": {
    "batch_size": 100,
    "timeout": 30,
    "retry_attempts": 3
  },
  "logging": {
    "level": "INFO",
    "file": "application.log"
  }
}
EOF
```

## Step 3: Index Your Documents

Now let's index these documents. There are two ways to do this:

### Option A: Using Claude Desktop (MCP Integration)

If you have Claude Desktop configured with EOL RAG Context:

1. Open Claude Desktop
2. Start a new conversation
3. Use the indexing tool:

```
Can you index the sample-project directory for me?
```

Claude will use the MCP tools to index your documents automatically.

### Option B: Using Python API

Create a simple indexing script:

```python
# index_sample.py
import asyncio
from pathlib import Path
from eol.rag_context import EOLRAGContextServer

async def index_documents():
    """Index the sample project documents."""
    
    # Initialize the server
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Index the current directory
    project_path = Path(".")
    print(f"Indexing documents in: {project_path.absolute()}")
    
    result = await server.index_directory(
        directory_path=str(project_path),
        recursive=True,
        file_patterns=["*.py", "*.md", "*.json"],
        force_reindex=True
    )
    
    print(f"Indexing Results:")
    print(f"  Indexed files: {result.get('indexed_files', 0)}")
    print(f"  Total chunks: {result.get('total_chunks', 0)}")
    print(f"  Processing time: {result.get('processing_time_seconds', 0):.2f}s")
    
    return result

if __name__ == "__main__":
    asyncio.run(index_documents())
```

Run the indexing script:
```bash
python index_sample.py
```

Expected output:
```
Indexing documents in: /path/to/sample-project
Indexing Results:
  Indexed files: 3
  Total chunks: 8
  Processing time: 1.23s
```

## Step 4: Search Your Documents

Now let's search the indexed content:

### Using Claude Desktop

Ask Claude questions about your code:

```
What does the DataProcessor class do?
```

```
How do I use the process_data method?
```

```
What configuration options are available?
```

### Using Python API

Create a search script:

```python
# search_sample.py
import asyncio
from eol.rag_context import EOLRAGContextServer
from eol.rag_context.server import SearchContextRequest

async def search_documents():
    """Search the indexed documents."""
    
    # Initialize the server
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Define search queries
    queries = [
        "What does the DataProcessor class do?",
        "How do I process data?",
        "What configuration options are available?",
        "Show me the main function"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        print("=" * 50)
        
        # Create search request
        request = SearchContextRequest(
            query=query,
            max_results=3,
            similarity_threshold=0.7
        )
        
        # Perform search
        results = await server.search_context(request, None)
        
        if results and results.get('results'):
            for i, result in enumerate(results['results'], 1):
                print(f"\n📄 Result {i}:")
                print(f"   File: {result['source_path']}")
                print(f"   Relevance: {result['similarity']:.3f}")
                print(f"   Content: {result['content'][:200]}...")
        else:
            print("   No results found")

if __name__ == "__main__":
    asyncio.run(search_documents())
```

Run the search script:
```bash
python search_sample.py
```

Expected output:
```
🔍 Query: What does the DataProcessor class do?
==================================================

📄 Result 1:
   File: main.py
   Relevance: 0.892
   Content: class DataProcessor:
    """A sample data processing class."""
    
    def __init__(self, name: str):
        """Initialize the processor with a name."""
        self.name = name
        self.processed_count = 0...

📄 Result 2:
   File: README.md
   Relevance: 0.845
   Content: - **Data Processing**: Process lists of strings
- **Length Calculation**: Calculate string lengths
- **Result Storage**: Store results in dictionaries...
```

## Step 5: Explore Advanced Features

### Hierarchical Search

Search at different levels of granularity:

```python
# Hierarchical search example
async def hierarchical_search():
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Search at concept level (high-level topics)
    concept_results = await server.search_context(SearchContextRequest(
        query="data processing architecture",
        max_results=2,
        search_level="concept"
    ), None)
    
    # Search at section level (specific sections)
    section_results = await server.search_context(SearchContextRequest(
        query="DataProcessor class methods",
        search_level="section"
    ), None)
    
    # Search at chunk level (detailed content)
    chunk_results = await server.search_context(SearchContextRequest(
        query="process_data method implementation",
        search_level="chunk"
    ), None)
```

### Knowledge Graph Exploration

Explore relationships between concepts:

```python
from eol.rag_context.server import QueryKnowledgeGraphRequest

async def explore_knowledge_graph():
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Query the knowledge graph
    kg_request = QueryKnowledgeGraphRequest(
        query="DataProcessor relationships",
        max_depth=2,
        include_relationships=True
    )
    
    kg_results = await server.query_knowledge_graph(kg_request, None)
    
    print("Knowledge Graph Results:")
    for entity in kg_results.get('entities', []):
        print(f"  Entity: {entity['name']} ({entity['type']})")
        for rel in entity.get('relationships', []):
            print(f"    → {rel['target']} ({rel['type']})")
```

### Real-time Updates

Set up file watching for automatic reindexing:

```python
async def setup_file_watching():
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Enable file watching
    watch_result = await server.start_file_watching(
        directory_path=".",
        patterns=["*.py", "*.md"],
        debounce_seconds=2.0
    )
    
    print("File watching enabled. Changes will trigger automatic reindexing.")
    print(f"Watching {len(watch_result.get('watched_paths', []))} paths")
    
    # Keep the server running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await server.stop_file_watching()
        print("File watching stopped.")
```

## Understanding the Results

### Search Result Structure

Each search result contains:

```python
{
    'content': str,           # The matched content
    'source_path': str,       # File path
    'chunk_type': str,        # Type of chunk (function, class, paragraph, etc.)
    'similarity': float,      # Similarity score (0.0-1.0)
    'metadata': {
        'lines': int,         # Line numbers in source file
        'language': str,      # Programming language (for code)
        'section': str,       # Section/header name
        'chunk_index': int    # Position in document
    }
}
```

### Similarity Scores

Understanding relevance scores:
- **0.9-1.0**: Exact or near-exact match
- **0.8-0.9**: Highly relevant content
- **0.7-0.8**: Moderately relevant
- **0.6-0.7**: Somewhat relevant
- **Below 0.6**: May not be relevant

### Content Types

Different types of chunks are indexed:
- **Functions/Methods**: Code functions and class methods
- **Classes**: Class definitions and docstrings
- **Sections**: Markdown headers and content blocks
- **Paragraphs**: Natural text paragraphs
- **Configurations**: JSON/YAML configuration blocks

## Performance Tips

### For Better Search Results

1. **Use Specific Queries**: "How to initialize DataProcessor" vs "initialization"
2. **Include Context**: "Python class methods" vs just "methods"
3. **Try Different Phrasings**: If no results, rephrase your question

### For Faster Indexing

1. **Filter File Types**: Only index relevant file extensions
2. **Set Size Limits**: Skip very large files that might not be useful
3. **Use Exclusion Patterns**: Skip build artifacts, logs, etc.

```python
# Optimized indexing configuration
indexing_config = {
    "file_patterns": ["*.py", "*.md", "*.rst"],  # Only relevant types
    "exclude_patterns": ["*.pyc", "__pycache__/*", "*.log"],
    "max_file_size_mb": 10,  # Skip large files
    "parse_code_structure": True,  # Better code understanding
}
```

## Troubleshooting

### Common Issues

**No search results found:**
- Check if indexing completed successfully
- Try broader search terms
- Lower the similarity threshold
- Verify files were included in indexing patterns

**Slow search performance:**
- Enable semantic caching
- Increase Redis connection pool size
- Use more specific queries

**Out of memory during indexing:**
- Reduce batch size in embedding configuration
- Process files in smaller batches
- Exclude large binary files

### Getting Help

**Check server logs:**
```bash
# If running with logging enabled
tail -f eol-rag-context.log
```

**Diagnostic commands:**
```bash
# Test configuration
eol-rag-context diagnose --config config.yaml

# Check Redis connection
eol-rag-context diagnose --redis-only

# Test embedding provider
eol-rag-context diagnose --embedding-only
```

**Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Next Steps

Now that you've completed your first indexing and search:

1. **[User Guide](../user-guide/)** - Learn about advanced indexing strategies
2. **[Configuration](configuration.md)** - Optimize settings for your use case  
3. **[API Reference](../api-reference/)** - Explore the complete API
4. **[Examples](../examples/)** - See real-world integration patterns

### Integration with Claude Desktop

To use this with Claude Desktop, add the MCP server configuration to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "eol-rag-context": {
      "command": "eol-rag-context",
      "args": ["serve", "--config", "/path/to/your/config.yaml"],
      "env": {
        "EOL_REDIS_URL": "redis://localhost:6379"
      }
    }
  }
}
```

Then restart Claude Desktop and you'll have access to intelligent context retrieval for all your conversations!