# Configuration

EOL Framework can be configured through environment variables, configuration files, or programmatically.

## Configuration Methods

### Environment Variables

The simplest way to configure EOL:

```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
export EMBEDDING_PROVIDER=sentence-transformers
export EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Configuration File

Create `eol.yaml` or `eol.json`:

```yaml
# eol.yaml
redis:
  host: localhost
  port: 6379
  db: 0
  password: null
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

### Programmatic Configuration

```python
from eol.rag_context import RAGConfig

config = RAGConfig(
    redis_host="localhost",
    redis_port=6379,
    embedding_provider="sentence-transformers",
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=500,
    chunk_overlap=50
)

# Use config in your application
indexer = DocumentIndexer(config=config)
```

## Configuration Reference

### Redis Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| host | `REDIS_HOST` | localhost | Redis server hostname |
| port | `REDIS_PORT` | 6379 | Redis server port |
| db | `REDIS_DB` | 0 | Redis database number |
| password | `REDIS_PASSWORD` | None | Redis password (if required) |
| max_connections | `REDIS_MAX_CONNECTIONS` | 10 | Maximum connection pool size |

### Embedding Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| provider | `EMBEDDING_PROVIDER` | sentence-transformers | Embedding provider |
| model | `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Model name |
| dimension | `EMBEDDING_DIMENSION` | 384 | Vector dimension |
| batch_size | `EMBEDDING_BATCH_SIZE` | 32 | Batch processing size |

### Processing Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| chunk_size | `CHUNK_SIZE` | 500 | Maximum chunk size |
| chunk_overlap | `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| chunk_strategy | `CHUNK_STRATEGY` | fixed | Chunking strategy |

## Advanced Configuration

### Custom Embedding Providers

```python
from eol.rag_context import EmbeddingManager

# OpenAI embeddings
manager = EmbeddingManager(
    provider="openai",
    api_key="your-api-key",
    model="text-embedding-ada-002"
)

# Hugging Face embeddings
manager = EmbeddingManager(
    provider="huggingface",
    model="BAAI/bge-small-en-v1.5"
)
```

### Performance Tuning

```yaml
performance:
  indexing:
    batch_size: 100
    max_workers: 4
    queue_size: 1000
  
  search:
    timeout: 30
    max_results: 100
    min_score: 0.5
  
  cache:
    max_size: 10000
    eviction_policy: lru
```

## Best Practices

1. **Use environment variables** for sensitive data (API keys, passwords)
2. **Version control** your configuration files (except secrets)
3. **Profile your workload** to optimize batch sizes
4. **Monitor performance** and adjust settings accordingly

---

Next: [First Steps](first-steps.md)