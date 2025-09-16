# Configuration Guide

The EOL RAG Context server can be configured through multiple methods, in order of precedence:

1. Command-line arguments
2. Environment variables
3. Configuration file
4. Default values

## Configuration File

The server looks for configuration in these locations:

1. Path specified on command line: `uv run eol-rag-mcp /path/to/config.yaml`
2. Project directory: `./config.yaml`
3. User config: `~/.config/eol-rag/config.yaml`

### Example Configuration

```yaml
# Redis connection
redis:
  host: localhost
  port: 6379
  password: ""  # Optional
  db: 0

# Embedding settings
embedding:
  provider: sentence-transformers  # or "openai"
  model_name: all-MiniLM-L6-v2
  dimension: 384
  cache_embeddings: true

# Semantic cache
cache:
  enabled: true
  ttl_seconds: 3600
  similarity_threshold: 0.97
  target_hit_rate: 0.31

# Context retrieval
context:
  max_context_tokens: 32000
  default_top_k: 10
  min_relevance_score: 0.7
  use_hierarchical_retrieval: true

# Document processing
document:
  chunk_size: 1000
  chunk_overlap: 200
  file_patterns:
    - "*.py"
    - "*.md"
    - "*.txt"
    - "*.js"
    - "*.ts"
  exclude_patterns:
    - "*.pyc"
    - "__pycache__/*"
    - ".git/*"
    - "node_modules/*"

# File watching
watcher:
  enabled: false
  debounce_seconds: 2.0
  use_polling: false  # Use polling instead of native events
```

## Environment Variables

All configuration options can be set via environment variables:

```bash
# Redis
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_PASSWORD=secret
export REDIS_DB=0

# Embeddings
export EMBEDDING_PROVIDER=sentence-transformers
export EMBEDDING_MODEL=all-MiniLM-L6-v2
export OPENAI_API_KEY=sk-...  # If using OpenAI

# Cache
export CACHE_ENABLED=true
export CACHE_TTL=3600
export CACHE_SIMILARITY_THRESHOLD=0.97

# Context
export MAX_CONTEXT_TOKENS=32000
export DEFAULT_TOP_K=10
export MIN_RELEVANCE_SCORE=0.7
```

## Configuration Options Reference

### Redis Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `redis.host` | string | "localhost" | Redis server hostname |
| `redis.port` | int | 6379 | Redis server port |
| `redis.password` | string | "" | Redis password (optional) |
| `redis.db` | int | 0 | Redis database number |

### Embedding Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `embedding.provider` | string | "sentence-transformers" | Embedding provider ("sentence-transformers" or "openai") |
| `embedding.model_name` | string | "all-MiniLM-L6-v2" | Model name for embeddings |
| `embedding.dimension` | int | 384 | Vector dimension (must match model) |
| `embedding.cache_embeddings` | bool | true | Cache computed embeddings |

### Cache Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `cache.enabled` | bool | true | Enable semantic caching |
| `cache.ttl_seconds` | int | 3600 | Cache entry time-to-live |
| `cache.similarity_threshold` | float | 0.97 | Similarity threshold for cache hits |
| `cache.target_hit_rate` | float | 0.31 | Target cache hit rate for optimization |

### Context Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `context.max_context_tokens` | int | 32000 | Maximum tokens in context window |
| `context.default_top_k` | int | 10 | Default number of results to return |
| `context.min_relevance_score` | float | 0.7 | Minimum relevance score for results |
| `context.use_hierarchical_retrieval` | bool | true | Use 3-level hierarchical search |

### Document Processing Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `document.chunk_size` | int | 1000 | Maximum chunk size in characters |
| `document.chunk_overlap` | int | 200 | Overlap between chunks |
| `document.file_patterns` | list | ["*.py", "*.md", "*.txt"] | File patterns to include |
| `document.exclude_patterns` | list | ["*.pyc", "__pycache__/*"] | Patterns to exclude |

### File Watcher Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `watcher.enabled` | bool | false | Enable automatic file watching |
| `watcher.debounce_seconds` | float | 2.0 | Seconds to wait before processing changes |
| `watcher.use_polling` | bool | false | Use polling instead of native file events |

## Claude Code Integration

When using with Claude Code, you can pass configuration through the MCP server setup:

```json
{
  "mcpServers": {
    "eol-rag-context": {
      "command": "uv",
      "args": ["run", "eol-rag-mcp", "~/.config/eol-rag/config.yaml"],
      "cwd": "/path/to/eol/packages/eol-rag-context",
      "env": {
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "CACHE_ENABLED": "true"
      }
    }
  }
}
```

## Common Configurations

### Development Setup

```yaml
redis:
  host: localhost
  port: 6379

embedding:
  provider: sentence-transformers
  model_name: all-MiniLM-L6-v2

cache:
  enabled: true
  ttl_seconds: 600  # Shorter TTL for development

document:
  file_patterns:
    - "*.py"
    - "*.md"
```

### Production Setup

```yaml
redis:
  host: redis.production.internal
  port: 6379
  password: ${REDIS_PASSWORD}  # From environment

embedding:
  provider: openai
  model_name: text-embedding-3-small

cache:
  enabled: true
  ttl_seconds: 7200
  similarity_threshold: 0.98  # Stricter for production

context:
  max_context_tokens: 50000  # Larger context for production
  min_relevance_score: 0.75
```

### Testing Setup

```yaml
redis:
  host: localhost
  port: 6379
  db: 15  # Separate DB for tests

cache:
  enabled: false  # Disable cache for deterministic tests

document:
  chunk_size: 500  # Smaller chunks for faster tests
```

# Vector Quantization Configuration Guide

## Overview

EOL RAG Context supports configurable vector quantization to optimize the trade-off between memory usage and search accuracy. With Redis 8.2+ Vector Sets, you can choose different quantization levels for different features of your application.

## Quantization Modes

### Q8 (8-bit Integer Quantization) - Default

- __Memory Usage__: 384 bytes per vector (75% reduction from float32)
- __Accuracy__: ~98-99% of original precision
- __Use Case__: Balanced choice for most applications
- __Performance__: Fast similarity computations

### NOQUANT (No Quantization)

- __Memory Usage__: 1,536 bytes per vector (full float32)
- __Accuracy__: 100% - no precision loss
- __Use Case__: Critical applications requiring maximum accuracy
- __Performance__: Highest quality results but more memory intensive

### BIN (Binary Quantization)

- __Memory Usage__: 48 bytes per vector (97% reduction)
- __Accuracy__: ~85-90% of original precision
- __Use Case__: Large-scale deployments where memory is critical
- __Performance__: Extremely fast but with reduced accuracy

## Configuration Options

### Global Default

Set the default quantization for all operations:

```python
from eol.rag_context.config import IndexConfig

config = IndexConfig(
    quantization="Q8"  # Global default
)
```

### Per-Hierarchy Level Configuration

Optimize quantization for each document hierarchy level:

```python
config = IndexConfig(
    # Global default
    quantization="Q8",

    # Hierarchy-specific settings
    concept_quantization="NOQUANT",  # High precision for concepts
    section_quantization="Q8",       # Balanced for sections
    chunk_quantization="BIN"         # Space-efficient for chunks
)
```

### Feature-Specific Configuration

Configure quantization for specific features:

```python
config = IndexConfig(
    quantization="Q8",

    # Semantic cache - optimize for space
    cache_quantization="BIN",

    # Batch operations - optimize for throughput
    batch_quantization="Q8"
)
```

### Environment Variables

Override settings via environment variables:

```bash
# Global setting
export INDEX_QUANTIZATION=NOQUANT

# Feature-specific
export INDEX_CONCEPT_QUANTIZATION=NOQUANT
export INDEX_SECTION_QUANTIZATION=Q8
export INDEX_CHUNK_QUANTIZATION=BIN
export INDEX_CACHE_QUANTIZATION=BIN
export INDEX_BATCH_QUANTIZATION=Q8
```

## Memory Impact Analysis

For 1 million 384-dimensional vectors:

| Quantization | Memory per Vector | Total Memory | Savings vs Float32 |
|--------------|------------------|--------------|-------------------|
| NOQUANT      | 1,536 bytes      | 1.5 GB       | 0%                |
| Q8           | 384 bytes        | 366 MB       | 75%               |
| BIN          | 48 bytes         | 46 MB        | 97%               |

## Best Practices

### 1. Hierarchy-Based Strategy

Different document levels have different accuracy requirements:

```python
config = IndexConfig(
    # Concepts: Fewer documents, need high precision
    concept_quantization="NOQUANT",

    # Sections: Moderate count, balanced needs
    section_quantization="Q8",

    # Chunks: Many documents, optimize for space
    chunk_quantization="BIN"
)
```

### 2. Use Case Optimization

#### High-Accuracy Search Applications

```python
config = IndexConfig(
    quantization="NOQUANT",
    cache_quantization="Q8"  # Cache can use less precision
)
```

#### Large-Scale Document Collections

```python
config = IndexConfig(
    quantization="Q8",
    chunk_quantization="BIN",  # Optimize chunks for space
    cache_quantization="BIN"    # Aggressive caching optimization
)
```

#### Balanced Performance

```python
config = IndexConfig(
    quantization="Q8"  # Use default for everything
)
```

### 3. Testing Quantization Impact

Monitor the impact of quantization on your specific use case:

```python
import time
import numpy as np
from eol.rag_context import EOLRAGContextServer

async def test_quantization_impact():
    # Test with different quantization settings
    configs = [
        ("NOQUANT", IndexConfig(quantization="NOQUANT")),
        ("Q8", IndexConfig(quantization="Q8")),
        ("BIN", IndexConfig(quantization="BIN"))
    ]

    for name, config in configs:
        server = EOLRAGContextServer(index_config=config)

        # Index test documents
        start = time.time()
        await server.index_folder("./test_docs")
        index_time = time.time() - start

        # Perform searches
        queries = ["test query 1", "test query 2", "test query 3"]
        search_times = []
        results_quality = []

        for query in queries:
            start = time.time()
            results = await server.search(query)
            search_times.append(time.time() - start)
            results_quality.append(len(results))

        print(f"\n{name} Quantization:")
        print(f"  Index Time: {index_time:.2f}s")
        print(f"  Avg Search Time: {np.mean(search_times):.3f}s")
        print(f"  Avg Results: {np.mean(results_quality):.1f}")
```

## Migration Guide

### From Hardcoded Q8

If you're upgrading from a version with hardcoded Q8 quantization:

1. __No Action Required__: Default behavior remains Q8
2. __Optional Optimization__: Configure per-feature quantization for better performance

### Changing Quantization Mode

__Warning__: Changing quantization requires re-indexing documents.

```python
# Step 1: Clear existing index
await server.clear_index()

# Step 2: Configure new quantization
config = IndexConfig(quantization="NOQUANT")

# Step 3: Re-index documents
await server.index_folder("./documents")
```

## Performance Considerations

### Q8 Quantization (Recommended)

- __Pros__: 75% memory savings, minimal accuracy loss
- __Cons__: Slight precision reduction
- __Best For__: Most production applications

### NOQUANT

- __Pros__: Maximum accuracy, no precision loss
- __Cons__: 4x memory usage vs Q8
- __Best For__: Financial data, medical records, legal documents

### BIN Quantization

- __Pros__: 97% memory savings, very fast
- __Cons__: Noticeable accuracy reduction
- __Best For__: Large-scale consumer applications, preliminary filtering

## Troubleshooting

### Issue: Reduced Search Quality with BIN

__Solution__: Use BIN only for chunks, keep concepts at Q8 or NOQUANT

### Issue: High Memory Usage

__Solution__: Enable quantization progressively:

1. Start with Q8 globally
2. Move chunks to BIN
3. Keep critical levels at Q8/NOQUANT

### Issue: Slow Indexing

__Solution__: Use Q8 for batch operations:

```python
config = IndexConfig(
    batch_quantization="Q8"  # Optimize batch indexing
)
```

## Advanced Configuration

### Custom Quantization Strategy

```python
class AdaptiveQuantizationStrategy:
    """Dynamically adjust quantization based on document importance."""

    def get_quantization_for_document(self, doc):
        if doc.metadata.get("priority") == "high":
            return "NOQUANT"
        elif doc.content_length > 10000:
            return "BIN"  # Large documents use binary
        else:
            return "Q8"  # Default
```

### Monitoring Quantization Impact

```python
from eol.rag_context.monitoring import QuantizationMonitor

monitor = QuantizationMonitor()
monitor.track_memory_usage()
monitor.track_search_accuracy()
monitor.generate_report()
```

## Conclusion

Vector quantization provides powerful options for optimizing your RAG system:

- __Start with Q8__ as the default for balanced performance
- __Configure per-feature__ based on your specific needs
- __Monitor impact__ on search quality and memory usage
- __Adjust progressively__ to find optimal settings

For most applications, the default Q8 quantization provides excellent balance. Use feature-specific configuration when you need to optimize for particular use cases.
