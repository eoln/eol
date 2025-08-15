# Installation Guide

## Prerequisites

### System Requirements

- **Python 3.11+** (required)
- **Redis Stack Server** (required for vector search)
- **macOS/Linux** (Windows support via WSL)

### Install System Dependencies

#### macOS (Homebrew)

```bash
# Install Python 3.11+
brew install python@3.11

# Install Redis Stack (includes RediSearch module)
brew tap redis-stack/redis-stack
brew install --cask redis-stack-server

# Install libmagic (for file type detection)
brew install libmagic

# Optional: Install uv for faster package management
brew install uv
```

#### Linux (Ubuntu/Debian)

```bash
# Install Python 3.11+
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip

# Install Redis Stack via Docker
docker run -d -p 6379:6379 redis/redis-stack:latest

# Install libmagic
sudo apt install libmagic1

# Optional: Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation Methods

### Method 1: Using uv (Recommended - Ultra Fast)

[uv](https://github.com/astral-sh/uv) is a blazing-fast Python package manager written in Rust.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or install with dev dependencies
uv pip install -e ".[dev]"
```

### Method 2: Using pip with pyproject.toml

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"

# Or install with all optional dependencies
pip install -e ".[all]"
```

### Method 3: Using pip with requirements.txt

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### Method 4: Using Make (Convenience Commands)

```bash
# Install package
make install

# Install with dev dependencies
make install-dev

# Install using uv (creates venv automatically)
make install-uv

# Complete development setup
make dev
```

## Dependency Groups

The project uses `pyproject.toml` to manage dependencies in groups:

### Core Dependencies (Always Installed)

- Redis client and numpy for vector operations
- Pydantic for configuration management
- FastMCP for MCP server
- Document processing libraries (BeautifulSoup, markdown, PyPDF, etc.)
- Tree-sitter for code parsing
- NetworkX for knowledge graphs

### Optional Dependencies

#### Development (`dev`)

```bash
pip install -e ".[dev]"
```

Includes: pytest, black, ruff, mypy, coverage tools

#### Testing (`test`)

```bash
pip install -e ".[test]"
```

Includes: pytest and testing utilities

#### Local Embeddings (`embeddings-local`)

```bash
pip install -e ".[embeddings-local]"
```

Includes: sentence-transformers for local embedding generation

#### OpenAI Embeddings (`embeddings-openai`)

```bash
pip install -e ".[embeddings-openai]"
```

Includes: openai client for OpenAI embeddings

#### All Optional Dependencies

```bash
pip install -e ".[all]"
```

## Setting Up for Development

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/eoln/eol-rag-context.git
cd eol-rag-context

# Run automated setup
./setup_test_environment.sh

# Or use Make
make dev
```

### Manual Setup

1. **Install dependencies:**

```bash
make install-dev
# or
pip install -e ".[dev]"
```

2. **Start Redis Stack:**

```bash
make redis-start
# or
redis-stack-server --daemonize yes
```

3. **Run tests to verify:**

```bash
make test
# or
pytest tests/
```

## Verifying Installation

### Check Dependencies

```bash
# Verify Python version
python --version  # Should be 3.11+

# Verify Redis with RediSearch
redis-cli MODULE LIST | grep search  # Should show search module

# Verify package installation
python -c "import eol.rag_context; print('OK')"

# Run test suite
make test
```

### Quick Test

```python
# test_install.py
from eol.rag_context import config, redis_client

# Test configuration
cfg = config.RAGConfig()
print(f"RAG Config: {cfg.project_root}")

# Test Redis connection (requires Redis running)
redis_cfg = config.RedisConfig()
store = redis_client.RedisVectorStore(redis_cfg, config.IndexConfig())
print("Installation successful!")
```

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError: No module named 'magic'

**Solution:** Install libmagic system dependency

```bash
# macOS
brew install libmagic

# Linux
sudo apt install libmagic1

# Then reinstall python-magic
pip install --force-reinstall python-magic
```

#### 2. Redis connection failed

**Solution:** Ensure Redis Stack is running

```bash
# Check if Redis is running
redis-cli ping

# Start Redis Stack
redis-stack-server --daemonize yes

# Or use Docker
docker run -d -p 6379:6379 redis/redis-stack:latest
```

#### 3. FT.CREATE command not found

**Solution:** You have regular Redis, not Redis Stack. Install Redis Stack:

```bash
# macOS
brew install --cask redis-stack-server

# Or use Docker
docker run -d -p 6379:6379 redis/redis-stack:latest
```

#### 4. Import errors with tree-sitter

**Solution:** Reinstall tree-sitter languages

```bash
pip install --force-reinstall tree-sitter tree-sitter-python tree-sitter-javascript
```

### Getting Help

1. Check the [Testing Guide](tests/integration/TESTING_GUIDE.md)
2. Run diagnostics: `python -m eol.rag_context.diagnostics`
3. File an issue: <https://github.com/eoln/eol-rag-context/issues>

## Docker Installation (Alternative)

For a containerized setup:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install with uv for speed
RUN pip install uv && \
    uv pip install -e .

CMD ["python", "-m", "eol.rag_context.server"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis/redis-stack:latest
    ports:
      - "6379:6379"
      - "8001:8001"  # RedisInsight

  eol-rag-context:
    build: .
    depends_on:
      - redis
    environment:
      REDIS_HOST: redis
      REDIS_PORT: 6379
    ports:
      - "8080:8080"
```

Run with:

```bash
docker-compose up
```

## Next Steps

After installation:

1. **Run tests:** `make test`
2. **Check coverage:** `make coverage`
3. **Start developing:** See [README.md](README.md)
4. **Read documentation:** See [docs/](docs/)
5. **Run examples:** See [examples/](examples/)
