# Vector Quantization Configuration Guide

## Overview

EOL RAG Context supports configurable vector quantization to optimize the trade-off between memory usage and search accuracy. With Redis 8.2+ Vector Sets, you can choose different quantization levels for different features of your application.

## Quantization Modes

### Q8 (8-bit Integer Quantization) - Default

- **Memory Usage**: 384 bytes per vector (75% reduction from float32)
- **Accuracy**: ~98-99% of original precision
- **Use Case**: Balanced choice for most applications
- **Performance**: Fast similarity computations

### NOQUANT (No Quantization)

- **Memory Usage**: 1,536 bytes per vector (full float32)
- **Accuracy**: 100% - no precision loss
- **Use Case**: Critical applications requiring maximum accuracy
- **Performance**: Highest quality results but more memory intensive

### BIN (Binary Quantization)

- **Memory Usage**: 48 bytes per vector (97% reduction)
- **Accuracy**: ~85-90% of original precision
- **Use Case**: Large-scale deployments where memory is critical
- **Performance**: Extremely fast but with reduced accuracy

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

1. **No Action Required**: Default behavior remains Q8
2. **Optional Optimization**: Configure per-feature quantization for better performance

### Changing Quantization Mode

**Warning**: Changing quantization requires re-indexing documents.

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

- **Pros**: 75% memory savings, minimal accuracy loss
- **Cons**: Slight precision reduction
- **Best For**: Most production applications

### NOQUANT

- **Pros**: Maximum accuracy, no precision loss
- **Cons**: 4x memory usage vs Q8
- **Best For**: Financial data, medical records, legal documents

### BIN Quantization

- **Pros**: 97% memory savings, very fast
- **Cons**: Noticeable accuracy reduction
- **Best For**: Large-scale consumer applications, preliminary filtering

## Troubleshooting

### Issue: Reduced Search Quality with BIN

**Solution**: Use BIN only for chunks, keep concepts at Q8 or NOQUANT

### Issue: High Memory Usage

**Solution**: Enable quantization progressively:

1. Start with Q8 globally
2. Move chunks to BIN
3. Keep critical levels at Q8/NOQUANT

### Issue: Slow Indexing

**Solution**: Use Q8 for batch operations:

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

- **Start with Q8** as the default for balanced performance
- **Configure per-feature** based on your specific needs
- **Monitor impact** on search quality and memory usage
- **Adjust progressively** to find optimal settings

For most applications, the default Q8 quantization provides excellent balance. Use feature-specific configuration when you need to optimize for particular use cases.
