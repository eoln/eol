# Examples

Complete working examples for EOL RAG Context, from quick start scenarios to advanced production deployments. All examples include full code, configuration, and step-by-step explanations.

## Overview

This section provides practical, tested examples that you can copy, modify, and use in your projects:

- **[Basic Usage](basic-usage.md)** - Simple examples to get started quickly
- **[Advanced Usage](advanced-usage.md)** - Complex scenarios and optimizations
- **[Integration Examples](integration-examples.md)** - Real-world integration patterns
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## Quick Start Examples

### Index and Search in 5 Minutes

The simplest possible example to see EOL RAG Context in action:

```python
import asyncio
from eol.rag_context import EOLRAGContextServer

async def quick_start():
    # Initialize server with defaults
    server = EOLRAGContextServer()
    await server.initialize()

    # Index some documents
    result = await server.index_directory(
        directory_path="./my_project",
        file_patterns=["*.py", "*.md"],
        recursive=True
    )

    print(f"✅ Indexed {result['indexed_files']} files")

    # Search for information
    search_result = await server.search_context({
        'query': 'how to configure the database',
        'max_results': 3
    }, None)

    # Display results
    for i, result in enumerate(search_result['results'], 1):
        print(f"\n📄 Result {i}:")
        print(f"   File: {result['source_path']}")
        print(f"   Score: {result['similarity']:.3f}")
        print(f"   Preview: {result['content'][:100]}...")

# Run the example
asyncio.run(quick_start())
```

### Claude Desktop Integration

Connect with Claude Desktop in 3 steps:

**1. Install and Configure:**

```bash
pip install eol-rag-context
```

**2. Configure Claude Desktop:**

```json
{
  "mcpServers": {
    "eol-rag-context": {
      "command": "eol-rag-context",
      "args": ["serve"],
      "env": {
        "EOL_REDIS_URL": "redis://localhost:6379"
      }
    }
  }
}
```

**3. Use in Claude:**

```
Can you index the files in /path/to/my/project and then help me understand the authentication system?
```

## Example Categories

### By Complexity Level

#### 🟢 **Beginner Examples**

Perfect for first-time users:

- [Simple file indexing](basic-usage.md#single-file-indexing)
- [Basic search queries](basic-usage.md#simple-search)
- [Configuration basics](basic-usage.md#configuration-basics)

#### 🟡 **Intermediate Examples**

For users with some experience:

- [Batch processing](basic-usage.md#batch-processing)
- [Custom chunking strategies](advanced-usage.md#chunking-optimization)
- [Performance monitoring](advanced-usage.md#performance-monitoring)

#### 🔴 **Advanced Examples**

For production deployments:

- [Knowledge graph construction](advanced-usage.md#knowledge-graphs)
- [Custom embedding providers](advanced-usage.md#custom-providers)
- [High-availability setups](advanced-usage.md#production-scaling)

### By Use Case

#### 📚 **Documentation Management**

- [Technical documentation indexing](basic-usage.md#documentation-indexing)
- [API documentation search](integration-examples.md#api-docs)
- [Knowledge base creation](advanced-usage.md#knowledge-base)

#### 💻 **Code Analysis**

- [Codebase exploration](basic-usage.md#code-analysis)
- [Function and class discovery](integration-examples.md#code-search)
- [Dependency analysis](advanced-usage.md#dependency-mapping)

#### 🔬 **Research & Analytics**

- [Research paper indexing](basic-usage.md#research-documents)
- [Data analysis workflows](advanced-usage.md#analytics-pipeline)
- [Multi-format collections](integration-examples.md#mixed-content)

#### 👥 **Team Collaboration**

- [Shared knowledge bases](integration-examples.md#team-setup)
- [Real-time collaboration](advanced-usage.md#collaborative-editing)
- [Multi-user access control](integration-examples.md#access-control)

## Running the Examples

### Prerequisites

All examples assume you have:

```bash
# Install EOL RAG Context
pip install eol-rag-context

# Start Redis Stack (required)
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

# Or using Homebrew on macOS
brew services start redis-stack
```

### Test Data Setup

Many examples use sample data. Create a test project:

```bash
# Create sample project structure
mkdir -p test_project/{src,docs,config}

# Add some sample files
echo "# My Project\nThis is a test project for EOL RAG Context." > test_project/README.md

cat > test_project/src/auth.py << 'EOF'
"""
Authentication module for user management.
"""

class UserAuth:
    """Handle user authentication and authorization."""

    def __init__(self, database):
        self.db = database

    def authenticate(self, username, password):
        """Authenticate user with username/password."""
        user = self.db.get_user(username)
        return user and self.verify_password(password, user.password_hash)

    def verify_password(self, password, hash):
        """Verify password against stored hash."""
        # Implementation here
        pass
EOF

cat > test_project/docs/setup.md << 'EOF'
# Setup Guide

## Installation

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Configure the database connection in `config/database.yaml`:

```yaml
database:
  host: localhost
  port: 5432
  name: myapp
```

EOF

cat > test_project/config/database.yaml << 'EOF'
database:
  host: localhost
  port: 5432
  name: myapp
  pool_size: 10
  timeout: 30

redis:
  url: redis://localhost:6379
  db: 0
EOF

```

### Running Individual Examples

Each example file is self-contained:

```bash
# Run a basic example
python examples/basic_file_indexing.py

# Run with custom configuration
python examples/advanced_search.py --config my_config.yaml

# Run integration example
python examples/claude_integration.py
```

### Testing Examples

Verify examples work in your environment:

```python
import asyncio
from pathlib import Path

async def test_environment():
    """Test that EOL RAG Context is properly set up."""
    try:
        from eol.rag_context import EOLRAGContextServer
        server = EOLRAGContextServer()
        await server.initialize()

        # Test Redis connection
        await server.redis_client.ping()
        print("✅ Redis connection successful")

        # Test embedding generation
        embedding = await server.get_embedding("test text")
        print(f"✅ Embeddings working (dimension: {len(embedding)})")

        print("🎉 Environment setup complete!")

    except ImportError:
        print("❌ EOL RAG Context not installed: pip install eol-rag-context")
    except Exception as e:
        print(f"❌ Setup issue: {e}")

asyncio.run(test_environment())
```

## Example Templates

### Basic Script Template

Use this template for simple scripts:

```python
"""
Template for EOL RAG Context examples.
Modify the functions below for your specific use case.
"""

import asyncio
import logging
from pathlib import Path
from eol.rag_context import EOLRAGContextServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Main example function."""
    # Initialize server
    server = EOLRAGContextServer()
    await server.initialize()

    try:
        # Your code here
        logger.info("Starting example...")

        # Example: Index files
        result = await server.index_directory("./test_data")
        logger.info(f"Indexed {result['indexed_files']} files")

        # Example: Search
        search_result = await server.search_context({
            'query': 'your search query',
            'max_results': 5
        }, None)

        logger.info(f"Found {len(search_result['results'])} results")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise
    finally:
        # Cleanup
        await server.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration Template

Use this YAML template for configurations:

```yaml
# EOL RAG Context Configuration Template
# Copy and modify for your use case

redis:
  url: "redis://localhost:6379"
  db: 0

embedding:
  provider: "sentence_transformers"  # or "openai"
  model: "all-MiniLM-L6-v2"        # fast and efficient
  batch_size: 32

indexing:
  chunk_size: 1000
  chunk_overlap: 200
  use_semantic_chunking: true

caching:
  enabled: true
  ttl_seconds: 3600
  target_hit_rate: 0.31

context:
  max_context_size: 8000

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Getting Help with Examples

### Common Issues

**Import Errors:**

```bash
# Make sure EOL RAG Context is installed
pip install eol-rag-context

# Check installation
python -c "import eol.rag_context; print('✅ Installed')"
```

**Redis Connection Issues:**

```bash
# Check if Redis is running
redis-cli ping

# Start Redis Stack if needed
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```

**File Not Found Errors:**

- Make sure file paths in examples exist
- Use absolute paths if relative paths don't work
- Check file permissions

### Debugging Examples

Add debug logging to any example:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run your example with detailed logs
```

### Contributing Examples

Have a great example to share? We'd love to include it:

1. **Follow the template** - Use consistent structure and documentation
2. **Test thoroughly** - Ensure examples work on different systems
3. **Document assumptions** - List prerequisites and setup steps
4. **Add error handling** - Show how to handle common failure cases
5. **Submit a PR** - Add to the [examples repository](https://github.com/eoln/eol)

## Next Steps

Ready to dive into specific examples?

### Start with the basics

→ **[Basic Usage Examples](basic-usage.md)** - Simple, practical examples for common tasks

### Explore advanced features

→ **[Advanced Usage Examples](advanced-usage.md)** - Complex scenarios and optimizations

### See real-world integrations

→ **[Integration Examples](integration-examples.md)** - Production-ready integration patterns

### Solve problems

→ **[Troubleshooting Examples](troubleshooting.md)** - Debug common issues with working solutions

---

**Need help with a specific use case?** Check the [troubleshooting guide](troubleshooting.md) or [open an issue](https://github.com/eoln/eol/issues) with details about your scenario.
