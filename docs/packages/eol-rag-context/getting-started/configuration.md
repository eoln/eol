# Configuration

This guide covers all configuration options for EOL RAG Context, from basic setup to advanced tuning for production deployments.

## Quick Configuration

The simplest way to get started is with environment variables:

```bash
# Basic Redis configuration
export EOL_REDIS_URL="redis://localhost:6379"
export EOL_REDIS_DB=0

# Embedding provider (choose one)
export EOL_EMBEDDING_PROVIDER="sentence_transformers"
export EOL_EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Or use OpenAI
export EOL_EMBEDDING_PROVIDER="openai"
export OPENAI_API_KEY="your-api-key-here"

# Start the server
eol-rag-context serve
```

## Configuration File

For more complex setups, create a `config.yaml` file:

```yaml
# config.yaml
redis:
  url: "redis://localhost:6379"
  db: 0
  password: null
  max_connections: 100

embedding:
  provider: "sentence_transformers"
  model: "all-MiniLM-L6-v2"
  dimension: 384
  batch_size: 32
  cache_embeddings: true

indexing:
  chunk_size: 1000
  chunk_overlap: 200
  max_file_size_mb: 50
  file_patterns:
    - "*.py"
    - "*.md"
    - "*.txt"
    - "*.rst"
    - "*.pdf"
    - "*.docx"

caching:
  enabled: true
  ttl_seconds: 3600
  similarity_threshold: 0.95
  target_hit_rate: 0.31
  max_cache_size: 1000

context:
  max_context_size: 8000
  context_overlap: 200
  hierarchy_weight: 0.3
```

Then start with:

```bash
eol-rag-context serve --config config.yaml
```

## Configuration Sections

### Redis Configuration

Redis Stack 8+ is required for vector search capabilities.

```yaml
redis:
  url: "redis://localhost:6379"     # Redis connection URL
  db: 0                             # Database number (0-15)
  password: null                    # Password if required
  max_connections: 100              # Connection pool size
  socket_timeout: 30                # Socket timeout in seconds
  socket_connect_timeout: 10        # Connection timeout
  retry_on_timeout: true            # Retry on timeout
  health_check_interval: 30         # Health check interval
```

**Redis Stack Setup:**

```bash
# Using Docker
docker run -d --name redis-stack \
  -p 6379:6379 -p 8001:8001 \
  redis/redis-stack:latest

# Using Homebrew (macOS)
brew install redis-stack

# Using package manager (Ubuntu)
curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
sudo sh -c 'echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" > /etc/apt/sources.list.d/redis.list'
sudo apt-get update
sudo apt-get install redis-stack-server
```

### Embedding Configuration

Choose between local and cloud-based embedding providers.

**Sentence Transformers (Local):**

```yaml
embedding:
  provider: "sentence_transformers"
  model: "all-MiniLM-L6-v2"        # Fast, good quality
  # model: "all-mpnet-base-v2"     # Higher quality, slower
  dimension: 384                    # Model-specific
  batch_size: 32                   # Batch size for processing
  cache_embeddings: true           # Cache computed embeddings
  device: "auto"                   # "cpu", "cuda", or "auto"
```

**OpenAI (Cloud):**

```yaml
embedding:
  provider: "openai"
  model: "text-embedding-ada-002"  # OpenAI's latest model
  dimension: 1536                  # Fixed for Ada-002
  batch_size: 100                  # OpenAI batch limit
  api_key_env: "OPENAI_API_KEY"    # Environment variable name
  max_retries: 3                   # Retry on failures
  timeout: 30                      # Request timeout
```

**Popular Models Comparison:**

| Model | Provider | Dimension | Speed | Quality | Use Case |
|-------|----------|-----------|-------|---------|----------|
| all-MiniLM-L6-v2 | SentenceTransformers | 384 | Very Fast | Good | Development, small projects |
| all-mpnet-base-v2 | SentenceTransformers | 768 | Medium | Excellent | Balanced performance |
| text-embedding-ada-002 | OpenAI | 1536 | Fast | Excellent | Production, best quality |

### Indexing Configuration

Control how documents are processed and chunked.

```yaml
indexing:
  chunk_size: 1000                 # Maximum chunk size in characters
  chunk_overlap: 200               # Overlap between chunks
  max_file_size_mb: 50            # Skip files larger than this
  parse_code_structure: true       # Use AST for code files
  extract_metadata: true           # Extract file metadata

  # File patterns to include
  file_patterns:
    - "*.py"       # Python files
    - "*.js"       # JavaScript files
    - "*.ts"       # TypeScript files
    - "*.md"       # Markdown files
    - "*.rst"      # reStructuredText
    - "*.txt"      # Plain text
    - "*.pdf"      # PDF documents
    - "*.docx"     # Word documents
    - "*.json"     # JSON files
    - "*.yaml"     # YAML files
    - "*.yml"      # YAML files

  # Patterns to exclude
  exclude_patterns:
    - "*.pyc"
    - "__pycache__/*"
    - "node_modules/*"
    - ".git/*"
    - "*.log"

chunking:
  use_semantic_chunking: true      # Respect paragraph boundaries
  markdown_split_headers: true     # Split Markdown by headers
  code_chunk_by_function: true     # Split code by functions/classes
  code_max_lines: 50               # Max lines per code chunk
```

### Semantic Cache Configuration

Optimize query response times with intelligent caching.

```yaml
caching:
  enabled: true                    # Enable semantic caching
  ttl_seconds: 3600               # Cache entry lifetime (1 hour)
  similarity_threshold: 0.95       # Minimum similarity for cache hit
  target_hit_rate: 0.31           # Target hit rate (research-optimized)
  adaptive_threshold: true         # Auto-adjust threshold
  max_cache_size: 1000            # Maximum cached entries
  eviction_policy: "lru"          # Eviction strategy
```

**Hit Rate Optimization:**

- **Target 31%**: Research shows this is optimal for semantic queries
- **Adaptive Threshold**: Automatically adjusts similarity threshold
- **TTL Balance**: Longer TTL = more hits but stale data
- **Cache Size**: Larger cache = more hits but more memory

### Context Window Configuration

Manage how context is assembled and returned.

```yaml
context:
  max_context_size: 8000          # Maximum context window size
  context_overlap: 200            # Overlap between context chunks
  hierarchy_weight: 0.3           # Weight for hierarchical relevance
  include_metadata: true          # Include chunk metadata

  # Context assembly strategy
  assembly_strategy: "hierarchical"  # "flat" or "hierarchical"

  # Result ranking
  ranking:
    semantic_weight: 0.7          # Semantic similarity weight
    recency_weight: 0.2           # File modification recency
    hierarchy_weight: 0.1         # Document structure weight
```

## Environment Variables

All configuration options can be set via environment variables using the pattern `EOL_<SECTION>_<OPTION>`:

```bash
# Redis configuration
export EOL_REDIS_URL="redis://localhost:6379"
export EOL_REDIS_DB=0
export EOL_REDIS_PASSWORD="secret"

# Embedding configuration
export EOL_EMBEDDING_PROVIDER="openai"
export EOL_EMBEDDING_MODEL="text-embedding-ada-002"
export OPENAI_API_KEY="your-key-here"

# Indexing configuration
export EOL_INDEXING_CHUNK_SIZE=1500
export EOL_INDEXING_MAX_FILE_SIZE_MB=100

# Cache configuration
export EOL_CACHING_ENABLED=true
export EOL_CACHING_TARGET_HIT_RATE=0.31
```

## Production Configuration

Recommended settings for production deployments:

```yaml
# production-config.yaml
redis:
  url: "${REDIS_URL}"
  password: "${REDIS_PASSWORD}"
  max_connections: 200
  socket_timeout: 60
  retry_on_timeout: true

embedding:
  provider: "openai"
  model: "text-embedding-ada-002"
  batch_size: 100
  max_retries: 5
  timeout: 60

indexing:
  chunk_size: 800                 # Smaller chunks for better precision
  chunk_overlap: 100              # Moderate overlap
  max_file_size_mb: 25            # Conservative file size limit

caching:
  enabled: true
  ttl_seconds: 7200               # 2 hour TTL
  similarity_threshold: 0.97      # Higher threshold for production
  max_cache_size: 5000            # Larger cache

context:
  max_context_size: 6000          # Conservative context size
  hierarchy_weight: 0.4           # Higher hierarchy weight

# Monitoring
logging:
  level: "INFO"
  format: "json"
  file: "/var/log/eol-rag-context.log"

metrics:
  enabled: true
  port: 9090
  path: "/metrics"
```

## Configuration Validation

The server validates configuration on startup and provides helpful error messages:

```bash
$ eol-rag-context serve --config invalid-config.yaml

Error: Invalid configuration
- redis.max_connections must be between 1 and 1000
- embedding.dimension must match model dimension (384 for all-MiniLM-L6-v2)
- indexing.chunk_size must be between 100 and 10000
- caching.similarity_threshold must be between 0.8 and 1.0

Use --validate-only to check configuration without starting server.
```

**Validation Command:**

```bash
# Check configuration without starting
eol-rag-context serve --config config.yaml --validate-only
```

## Performance Tuning

### Memory Optimization

```yaml
embedding:
  batch_size: 16                  # Reduce for limited memory
  cache_embeddings: false         # Disable if memory constrained

indexing:
  chunk_size: 500                 # Smaller chunks = less memory

caching:
  max_cache_size: 100             # Reduce cache size
```

### Speed Optimization

```yaml
embedding:
  batch_size: 64                  # Increase batch size
  device: "cuda"                  # Use GPU if available

indexing:
  parse_code_structure: false     # Disable AST parsing

redis:
  max_connections: 200            # Increase connection pool
```

### Quality Optimization

```yaml
embedding:
  model: "all-mpnet-base-v2"      # Higher quality model

indexing:
  chunk_size: 1500                # Larger chunks for more context
  chunk_overlap: 300              # More overlap

caching:
  similarity_threshold: 0.98      # Higher precision threshold
```

## Troubleshooting

### Common Configuration Issues

**Redis Connection Failed:**

```
Error: Redis connection failed: Connection refused
```

- Verify Redis Stack is running: `redis-cli ping`
- Check URL format: `redis://host:port/db`
- Ensure Redis Stack (not regular Redis) for vector search

**Embedding Model Not Found:**

```
Error: Model 'invalid-model' not found
```

- Check available models: `eol-rag-context list-models`
- Verify model name spelling and provider

**Out of Memory:**

```
Error: CUDA out of memory
```

- Reduce batch size: `batch_size: 8`
- Switch to CPU: `device: "cpu"`
- Use smaller model: `model: "all-MiniLM-L6-v2"`

**Slow Indexing:**

```
Warning: Indexing taking longer than expected
```

- Reduce file size limit: `max_file_size_mb: 10`
- Disable code parsing: `parse_code_structure: false`
- Increase batch size: `batch_size: 64`

### Configuration Testing

Test your configuration with the built-in diagnostics:

```bash
# Test all components
eol-rag-context diagnose --config config.yaml

# Test specific components
eol-rag-context diagnose --redis-only
eol-rag-context diagnose --embedding-only
eol-rag-context diagnose --config-only
```

## Next Steps

With your configuration complete:

1. **[First Steps](first-steps.md)** - Index your first documents
2. **[User Guide](../user-guide/)** - Learn advanced features
3. **[API Reference](../api-reference/)** - Explore the full API
4. **[Examples](../examples/)** - See real-world usage patterns

For production deployments, see our [Deployment Guide](../development/deployment.md).
