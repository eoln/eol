# Parallel Indexing Architecture for Large Repositories

## Current Limitations

### Performance Bottlenecks
- **Sequential file processing**: Files processed one-by-one in main thread
- **Multiple async operations per file**: 4+ embeddings + 4+ Redis operations per file
- **No batch operations**: Individual Redis VADD commands for each document
- **Memory inefficient**: Full file content loaded for processing and hashing
- **No resumability**: Must restart from beginning if interrupted

### Scalability Issues for Large Repos (LOTR-scale)
- **Time complexity**: O(n) sequential with high constant factor (~2-5 seconds per file)
- **Memory usage**: Unbounded growth with large files and embedding caches
- **Error recovery**: Single file error can fail entire batch
- **Progress tracking**: Limited visibility into indexing progress

## Proposed Architecture: Multi-Level Parallelization

### 1. File Discovery & Batching Layer
```python
class ParallelFileScanner:
    async def scan_repository(self, path: Path) -> AsyncIterator[FileBatch]:
        """Stream file batches for processing"""
        # Intelligent batching by file size, type, and complexity
        # Priority queuing: critical files first
        # Skip patterns and filters applied early
```

### 2. Worker Pool Architecture
```python
class IndexingWorkerPool:
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.document_workers = asyncio.Semaphore(max_workers)
        self.embedding_workers = asyncio.Semaphore(max_workers // 2)  
        self.redis_workers = asyncio.Semaphore(max_workers // 4)
```

### 3. Pipeline Stages with Backpressure

#### Stage 1: Document Processing (CPU Intensive)
- **Parallel document parsing** with `asyncio.gather()`
- **Memory streaming** for large files
- **Intelligent chunking** based on content type
- **Error isolation** per file

#### Stage 2: Embedding Generation (I/O + Compute)
- **Batch embedding generation** (32+ texts per request)
- **Embedding cache** with LRU eviction
- **Request batching** to maximize throughput
- **Provider-specific optimizations** (OpenAI batch API, local model pooling)

#### Stage 3: Vector Set Operations (I/O Intensive) 
- **Redis pipelining** for batch VADD operations
- **Transaction grouping** for consistency
- **Connection pooling** for concurrent operations
- **Bulk operations** where possible

### 4. Progress Tracking & Resumability
```python
class IndexingCheckpoint:
    processed_files: Set[str]
    failed_files: Dict[str, str]  # file -> error
    batch_progress: Dict[str, float]
    total_files: int
    start_time: float
    
    def save_state(self) -> None:
        """Persist checkpoint to Redis"""
    
    def can_resume(self) -> bool:
        """Check if indexing can be resumed"""
```

### 5. Memory Management
- **Streaming file processing** for large files
- **Embedding cache with size limits**
- **Garbage collection hints** after processing batches
- **Memory pressure monitoring** with backpressure

## Implementation Strategy

### Phase 1: Batch Processing Foundation
1. **BatchDocumentProcessor**: Process multiple files concurrently
2. **BatchEmbeddingManager**: Generate embeddings in batches
3. **BatchRedisClient**: Use Redis pipelining for bulk operations

### Phase 2: Worker Pool Implementation
1. **IndexingWorkerPool**: Manage concurrent workers with backpressure
2. **FileDiscoveryStream**: Async iterator for file batches
3. **ProgressTracker**: Real-time progress updates and checkpointing

### Phase 3: Advanced Optimizations
1. **Intelligent file prioritization**: Index critical files first
2. **Content-aware batching**: Group similar files for processing efficiency
3. **Adaptive worker scaling**: Adjust worker count based on system load
4. **Memory pressure handling**: Graceful degradation under memory constraints

### Phase 4: Resumability & Reliability
1. **Checkpoint persistence**: Save/restore indexing state
2. **Error recovery**: Retry failed files with exponential backoff
3. **Partial indexing**: Continue from interruption point
4. **Health monitoring**: Track worker performance and errors

## Performance Improvements

### Expected Speedup (Conservative Estimates)
- **File Processing**: 8-16x speedup with parallel document processing
- **Embedding Generation**: 4-8x speedup with batch requests
- **Redis Operations**: 3-5x speedup with pipelining
- **Overall**: **10-30x speedup** for large repositories

### Memory Efficiency
- **Streaming processing**: Constant memory usage regardless of file size
- **Bounded caches**: LRU eviction prevents memory bloat
- **Batch processing**: Reduced Python object overhead

### Scalability Characteristics
- **Linear scaling**: Performance scales with available CPU/memory
- **Graceful degradation**: Adapts to system resource constraints
- **Large repository optimized**: Designed for 10k+ files, GB+ repositories

## API Design

### Backward Compatible Interface
```python
# Existing API still works
await indexer.index_folder("/huge/repo")

# New high-performance API
await indexer.index_folder_parallel(
    "/huge/repo",
    max_workers=16,
    batch_size=32,
    enable_checkpointing=True,
    progress_callback=callback
)
```

### Advanced Configuration
```python
parallel_config = ParallelIndexingConfig(
    max_document_workers=16,
    max_embedding_workers=8, 
    max_redis_workers=4,
    batch_size=32,
    enable_streaming=True,
    checkpoint_interval=100,
    memory_limit_mb=2048
)

await indexer.index_folder_parallel("/huge/repo", config=parallel_config)
```

## Integration with Vector Sets

### Optimized for Redis 8.2+ Vector Sets
- **Bulk VADD operations**: Multiple vectors per command
- **Connection multiplexing**: Concurrent Redis operations
- **Transaction batching**: Group related operations
- **Memory-efficient serialization**: Minimize data transfer

### Hierarchical Indexing Optimization
- **Concept extraction**: Parallel processing across files
- **Section processing**: Batch similar content types
- **Chunk operations**: Bulk Vector Set operations

## Monitoring & Observability

### Real-time Metrics
- Files/second processing rate
- Embedding generation throughput
- Redis operation latency
- Memory usage and GC pressure
- Worker utilization and queue depths

### Progress Reporting
```python
def progress_callback(status: IndexingStatus):
    print(f"Progress: {status.completed_files}/{status.total_files} "
          f"({status.completion_percentage:.1f}%) "
          f"Rate: {status.files_per_second:.1f} files/sec "
          f"ETA: {status.estimated_completion}")
```

This architecture transforms the indexing from a sequential, blocking process into a high-performance, parallel pipeline capable of efficiently indexing massive repositories like LOTR while maintaining reliability and observability.