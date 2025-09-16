# Installation

This guide covers installing the EOL Framework and its dependencies.

## Prerequisites

### System Requirements

- **Python**: 3.13 or higher
- **Operating System**: Linux, macOS, or Windows (WSL2 recommended)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 1GB free space for dependencies

### Required Services

#### Redis Stack

EOL requires Redis with vector search capabilities:

=== "Docker (Recommended)"

    ```bash
    docker run -d \
      --name redis-stack \
      -p 6379:6379 \
      -p 8001:8001 \
      redis/redis-stack:latest
    ```

=== "Homebrew (macOS)"

    ```bash
    brew tap redis-stack/redis-stack
    brew install redis-stack
    redis-stack-server
    ```

=== "Native Installation"

    See [Redis Stack installation guide](https://redis.io/docs/stack/get-started/install/)

## Installation Methods

### Quick Install with uv (Recommended)

[uv](https://github.com/astral-sh/uv) is an ultra-fast Python package manager (10-100x faster than pip):

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install EOL RAG Context
uv pip install eol-rag-context
```

### Standard Installation with pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install eol-rag-context
```

### Development Installation

For contributing or development work:

```bash
# Clone the repository
git clone https://github.com/eoln/eol.git
cd eol

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e ".[dev]"
```

### Docker Installation

```bash
# Pull the Docker image
docker pull eoln/eol-rag-context:latest

# Run with docker-compose
docker-compose up -d
```

## Optional Dependencies

### Embedding Providers

Install additional providers based on your needs:

```bash
# OpenAI embeddings
uv pip install "eol-rag-context[openai]"

# Hugging Face embeddings
uv pip install "eol-rag-context[huggingface]"

# All providers
uv pip install "eol-rag-context[all]"
```

### Development Tools

For development and testing:

```bash
# Install development dependencies
uv pip install "eol-rag-context[dev]"

# This includes:
# - pytest and testing tools
# - Documentation tools (mkdocs)
# - Linting and formatting tools
# - Type checking tools
```

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Embedding Configuration (optional)
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2

# OpenAI Configuration (if using OpenAI)
OPENAI_API_KEY=your-api-key-here

# Performance Settings
BATCH_SIZE=100
MAX_WORKERS=4
```

### Configuration File

Create `config.yaml` for more complex configurations:

```yaml
redis:
  host: localhost
  port: 6379
  db: 0
  max_connections: 10

embeddings:
  provider: sentence-transformers
  model: all-MiniLM-L6-v2
  dimension: 384
  batch_size: 32

processing:
  chunk_size: 500
  chunk_overlap: 50
  chunk_strategy: semantic

cache:
  enabled: true
  ttl: 3600
  similarity_threshold: 0.95
```

## Verification

### Check Installation

```python
# Verify installation
python -c "import eol.rag_context; print(eol.rag_context.__version__)"
```

### Test Redis Connection

```python
from eol.rag_context import RedisStore

# Test connection
store = RedisStore()
store.connect()
print("Redis connection successful!")
```

### Run Quick Example

```python
from eol.rag_context import DocumentIndexer, RedisStore

# Initialize
redis_store = RedisStore()
redis_store.connect()

indexer = DocumentIndexer(redis_store)

# Index a simple document
doc = {
    "id": "test-1",
    "content": "EOL is a RAG framework for AI applications.",
    "metadata": {"source": "test"}
}

indexer.index_document(doc)
print("Document indexed successfully!")

# Search
results = redis_store.search_similar("What is EOL?", k=1)
print(f"Found: {results[0].content if results else 'No results'}")
```

## Troubleshooting

### Common Issues

#### Redis Connection Failed

```bash
# Check if Redis is running
redis-cli ping

# If not, start Redis
docker start redis-stack
# or
redis-stack-server
```

#### Python Version Error

```bash
# Check Python version
python --version

# If < 3.13, install newer version
# macOS: brew install python@3.13
# Ubuntu: sudo apt install python3.13
# Or use pyenv for version management
```

#### Missing Dependencies

```bash
# Reinstall with all dependencies
uv pip install --force-reinstall "eol-rag-context[all]"
```

### Getting Help

- Check the [FAQ](../packages/eol-rag-context/examples/troubleshooting.md)
- Search [GitHub Issues](https://github.com/eoln/eol/issues)
- Ask in [Discussions](https://github.com/eoln/eol/discussions)

## Next Steps

Now that you have EOL installed, you can:

- Follow the [Quick Start Guide](quickstart.md)
- Explore [Configuration Options](configuration.md)
- Read the [User Guide](../packages/eol-rag-context/user-guide/index.md)
- Check out [Examples](../packages/eol-rag-context/examples/index.md)

---

Need help? Join our [community discussions](https://github.com/eoln/eol/discussions) or [report an issue](https://github.com/eoln/eol/issues).
