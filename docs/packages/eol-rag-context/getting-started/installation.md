# Installation

This guide covers installing EOL RAG Context in different environments.

## Prerequisites

### System Requirements

- **Python**: 3.11 or higher
- **Memory**: 4GB+ RAM recommended
- **Storage**: 500MB+ free space for indexes
- **Network**: Internet access for embedding models (first run)

### Redis Stack 8.0+

Redis Stack includes the RediSearch module required for vector operations. Regular Redis will **not** work.

## Installation Methods

### Method 1: pip (Recommended)

```bash
pip install eol-rag-context
```

### Method 2: From Source

```bash
git clone https://github.com/eoln/eol.git
cd eol/packages/eol-rag-context
pip install -e .
```

## Redis Setup

### Docker (Easiest)

```bash
# Start Redis Stack container
docker run -d \
  --name redis-rag \
  -p 6379:6379 \
  -p 8001:8001 \
  redis/redis-stack:latest

# Verify Redis is running
redis-cli ping
# Should return: PONG
```

### Native Installation

**macOS (Homebrew):**

```bash
brew install redis-stack-server
redis-stack-server --daemonize yes
```

**Ubuntu/Debian:**

```bash
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update
sudo apt-get install redis-stack-server
```

## Verification

Test your installation:

```bash
# Check eol-rag-context is installed
eol-rag-context --version

# Test Redis Stack connection
redis-cli FT._LIST
# Should return list of indexes (empty initially)

# Test basic functionality
python -c "
from eol.rag_context import EOLRAGContextServer
server = EOLRAGContextServer()
print('✅ Installation successful!')
"
```

## Troubleshooting

### Common Issues

**"Command not found: eol-rag-context"**

- Make sure you're in the correct virtual environment
- Try `python -m eol.rag_context` instead

**"Cannot connect to Redis"**

- Verify Redis Stack is running: `docker ps | grep redis`
- Check port 6379 is available: `netstat -an | grep 6379`
- Ensure you're using Redis Stack, not regular Redis

**"No module named sentence_transformers"**

- This is normal for first run - the embedding model will download automatically
- Requires internet connection for initial model download

## Next Steps

Now that you have EOL RAG Context installed:

1. [Configure the system](configuration.md) for your environment
2. [Take your first steps](first-steps.md) with document indexing
3. Explore the [User Guide](../user-guide/) for advanced features
