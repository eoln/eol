# Document Indexing

This guide covers everything you need to know about indexing documents effectively with EOL RAG Context, from basic file processing to advanced optimization strategies.

## Overview

Document indexing is the process of converting your files into a searchable knowledge base. EOL RAG Context uses intelligent chunking strategies to break documents into semantically meaningful pieces while preserving context and relationships.

### Key Concepts

- **Hierarchical Organization**: 3-level structure (concepts → sections → chunks)
- **Format-Specific Processing**: Optimized strategies for each file type
- **Intelligent Chunking**: Structure-aware splitting preserves meaning
- **Vector Embeddings**: Content converted to searchable vectors
- **Metadata Extraction**: Rich context for enhanced retrieval

## Basic Indexing

### Single File Indexing

Start with indexing a single file to understand the process:

```python
import asyncio
from pathlib import Path
from eol.rag_context import EOLRAGContextServer

async def index_single_file():
    # Initialize server
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Index a single file
    result = await server.index_file(
        file_path="/path/to/document.py",
        force_reindex=True
    )
    
    print(f"Indexed: {result['file_path']}")
    print(f"Chunks created: {result['chunks_created']}")
    print(f"Processing time: {result['processing_time']:.2f}s")

asyncio.run(index_single_file())
```

### Directory Indexing

Index entire directories with flexible patterns:

```python
async def index_directory():
    server = EOLRAGContextServer()
    await server.initialize()
    
    result = await server.index_directory(
        directory_path="/path/to/project",
        recursive=True,
        file_patterns=["*.py", "*.md", "*.rst"],
        exclude_patterns=["*.pyc", "__pycache__/*", ".git/*"],
        force_reindex=False  # Skip unchanged files
    )
    
    print(f"Indexed {result['indexed_files']} files")
    print(f"Created {result['total_chunks']} chunks")
    print(f"Skipped {result['skipped_files']} unchanged files")

asyncio.run(index_directory())
```

## File Format Support

### Code Files

Code files use AST (Abstract Syntax Tree) parsing for structure-aware chunking.

**Supported Languages:**
- Python (`.py`)
- JavaScript/TypeScript (`.js`, `.jsx`, `.ts`, `.tsx`)
- Rust (`.rs`)
- Go (`.go`)
- Java (`.java`)
- C/C++ (`.c`, `.cpp`, `.h`)
- C# (`.cs`)

**Chunking Strategy:**
```python
# Python code is chunked by:
class MyClass:           # ← Class-level chunk
    def method1(self):   # ← Method-level chunk
        pass
    
    def method2(self):   # ← Separate method chunk
        pass

def standalone_function():  # ← Function-level chunk
    pass
```

**Configuration:**
```yaml
indexing:
  parse_code_structure: true    # Enable AST parsing
  code_chunk_by_function: true  # Chunk by functions/classes
  code_max_lines: 50           # Max lines per chunk fallback
```

### Markdown Files

Markdown files are processed with header-aware chunking:

```markdown
# Main Header               ← Concept level
## Section Header          ← Section level
Content paragraph...       ← Chunk level

### Subsection            ← Section level
More content...           ← Chunk level
```

**Configuration:**
```yaml
chunking:
  markdown_split_headers: true    # Split by headers
  use_semantic_chunking: true     # Respect paragraph boundaries
```

### PDF Documents

PDF processing extracts text while preserving document structure:

**Features:**
- Text extraction from pages
- Metadata extraction (title, author, creation date)
- Paragraph-based chunking
- Page number preservation

**Example:**
```python
async def index_pdf_collection():
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Configure for PDF processing
    config = {
        "max_file_size_mb": 50,  # Skip very large PDFs
        "extract_metadata": True,
        "chunk_by_pages": False,  # Use paragraph chunking
    }
    
    result = await server.index_directory(
        directory_path="/path/to/pdfs",
        file_patterns=["*.pdf"],
        **config
    )
```

### Structured Data

JSON and YAML files are processed with structure awareness:

```json
{
  "database": {           ← Object-level chunk
    "host": "localhost",
    "port": 5432
  },
  "features": [           ← Array-level chunk
    "auth", "logging"
  ]
}
```

**Configuration:**
```yaml
indexing:
  structured_chunk_by_key: true   # Chunk by top-level keys
  preserve_structure: true        # Maintain JSON/YAML format
```

## Advanced Chunking Strategies

### Semantic Chunking

Intelligent text splitting that respects natural boundaries:

```python
async def configure_semantic_chunking():
    config = {
        "chunking": {
            "use_semantic_chunking": True,
            "chunk_size": 1000,           # Characters per chunk
            "chunk_overlap": 200,         # Overlap for context
            "respect_boundaries": True,   # Don't split mid-sentence
            "paragraph_threshold": 0.3,   # Minimum paragraph coherence
        }
    }
    
    server = EOLRAGContextServer(config=config)
    await server.initialize()
```

**Benefits:**
- Preserves meaning across chunk boundaries
- Maintains paragraph coherence
- Better context for search results
- Improved vector representations

### Hierarchical Chunking

Create multiple granularity levels for different search needs:

```python
async def setup_hierarchical_indexing():
    config = {
        "indexing": {
            "create_hierarchy": True,
            "hierarchy_levels": {
                "concept": {          # High-level topics
                    "chunk_size": 2000,
                    "overlap": 100,
                },
                "section": {          # Medium-level sections  
                    "chunk_size": 1000,
                    "overlap": 200,
                },
                "chunk": {            # Detailed chunks
                    "chunk_size": 500,
                    "overlap": 100,
                }
            }
        }
    }
```

### Content-Aware Processing

Adjust processing based on content characteristics:

```python
async def content_aware_indexing():
    # Different strategies for different content types
    processing_rules = {
        "code_files": {
            "chunk_by": "ast_nodes",
            "preserve_scope": True,
            "include_comments": True,
        },
        "documentation": {
            "chunk_by": "headers",
            "maintain_hierarchy": True,
            "extract_examples": True,
        },
        "configuration": {
            "chunk_by": "sections",
            "preserve_structure": True,
            "validate_syntax": True,
        }
    }
    
    for content_type, rules in processing_rules.items():
        await server.index_with_rules(
            path="/path/to/content",
            content_type=content_type,
            **rules
        )
```

## Batch Processing

### Large Document Collections

Efficiently process thousands of documents:

```python
async def batch_index_large_collection():
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Configure for batch processing
    batch_config = {
        "batch_size": 50,           # Process 50 files at once
        "parallel_workers": 4,      # Use 4 parallel workers
        "checkpoint_interval": 100,  # Save progress every 100 files
        "error_handling": "skip",   # Skip problematic files
        "progress_callback": log_progress,
    }
    
    # Process large collection
    result = await server.batch_index(
        directories=[
            "/path/to/docs",
            "/path/to/code", 
            "/path/to/research"
        ],
        **batch_config
    )
    
    print(f"Processed {result['total_files']} files")
    print(f"Success rate: {result['success_rate']:.1%}")

def log_progress(completed, total, current_file):
    progress = (completed / total) * 100
    print(f"Progress: {progress:.1f}% - Processing: {current_file}")

asyncio.run(batch_index_large_collection())
```

### Incremental Updates

Only process changed files for faster updates:

```python
async def incremental_indexing():
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Index only changed files since last run
    result = await server.index_directory(
        directory_path="/path/to/project",
        incremental=True,           # Only process changes
        checksum_validation=True,   # Verify file changes
        track_deletions=True,       # Remove deleted files from index
    )
    
    print(f"Updated files: {result['updated_files']}")
    print(f"Deleted files: {result['deleted_files']}")
    print(f"Unchanged files: {result['unchanged_files']}")
```

## Performance Optimization

### Memory Management

Control memory usage during indexing:

```python
# Memory-efficient configuration
memory_config = {
    "embedding": {
        "batch_size": 16,           # Smaller batches
        "cache_embeddings": False,  # Don't cache in memory
        "streaming_mode": True,     # Process in streams
    },
    "indexing": {
        "chunk_buffer_size": 100,   # Buffer chunks before writing
        "gc_interval": 50,          # Run garbage collection
        "memory_limit_mb": 1024,    # Limit memory usage
    }
}
```

### Speed Optimization

Maximize indexing throughput:

```python
# High-performance configuration
speed_config = {
    "embedding": {
        "batch_size": 64,           # Larger batches
        "device": "cuda",           # Use GPU if available
        "parallel_embedding": True,  # Parallel embedding generation
    },
    "indexing": {
        "skip_binary_files": True,  # Don't process binaries
        "fast_text_detection": True, # Quick text file detection
        "parallel_processing": True, # Process multiple files
        "worker_count": 8,          # Number of worker threads
    },
    "redis": {
        "pipeline_size": 100,       # Batch Redis operations
        "connection_pool_size": 20, # More Redis connections
    }
}
```

### Quality vs Speed Tradeoffs

Balance processing quality with speed:

```python
# Development mode (fast, lower quality)
dev_config = {
    "chunking": {
        "use_semantic_chunking": False,  # Simple splitting
        "chunk_size": 2000,             # Larger chunks
    },
    "embedding": {
        "model": "all-MiniLM-L6-v2",    # Faster model
        "batch_size": 64,
    }
}

# Production mode (slower, higher quality)
prod_config = {
    "chunking": {
        "use_semantic_chunking": True,   # Smart splitting
        "chunk_size": 800,              # Smaller, focused chunks
        "overlap": 200,
    },
    "embedding": {
        "model": "all-mpnet-base-v2",    # Higher quality model
        "batch_size": 32,
    }
}
```

## Monitoring and Analytics

### Indexing Metrics

Track indexing performance and quality:

```python
async def monitor_indexing():
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Enable metrics collection
    await server.enable_metrics(
        track_performance=True,
        track_quality=True,
        export_metrics=True,
        metrics_interval=60  # seconds
    )
    
    # Get indexing statistics
    stats = await server.get_indexing_stats()
    
    print("Indexing Performance:")
    print(f"  Files per minute: {stats['files_per_minute']}")
    print(f"  Chunks per file: {stats['avg_chunks_per_file']}")
    print(f"  Processing time per MB: {stats['time_per_mb']:.2f}s")
    
    print("Quality Metrics:")
    print(f"  Average chunk size: {stats['avg_chunk_size']} chars")
    print(f"  Embedding quality score: {stats['embedding_quality']}")
    print(f"  Chunk coherence score: {stats['chunk_coherence']}")
```

### Health Monitoring

Monitor system health during indexing:

```python
async def health_monitoring():
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Set up health monitoring
    health_config = {
        "memory_threshold_mb": 2048,    # Alert if exceeding 2GB
        "processing_timeout": 300,      # Alert if file takes >5min
        "error_rate_threshold": 0.05,   # Alert if >5% files fail
        "redis_connection_timeout": 30,
    }
    
    await server.setup_health_monitoring(**health_config)
    
    # Check system health
    health = await server.get_health_status()
    
    if health['status'] == 'healthy':
        print("✅ System healthy")
    else:
        print("⚠️ Health issues:")
        for issue in health['issues']:
            print(f"   {issue['severity']}: {issue['message']}")
```

## File Pattern Strategies

### Smart Pattern Selection

Choose file patterns based on your use case:

```python
# Code-focused project
code_patterns = [
    "*.py", "*.js", "*.ts", "*.jsx", "*.tsx",
    "*.go", "*.rs", "*.java", "*.cpp", "*.c", "*.h",
    "*.md", "*.rst",  # Documentation
    "*.json", "*.yaml", "*.yml",  # Configuration
]

# Research/documentation project  
docs_patterns = [
    "*.md", "*.rst", "*.txt",
    "*.pdf", "*.docx",
    "*.tex", "*.org",  # Academic formats
]

# Mixed content project
mixed_patterns = [
    "*.py", "*.js", "*.md",     # Core files
    "*.pdf", "*.docx",          # Documents  
    "*.json", "*.yaml",         # Config
    "*.sql",                    # Database
]

# Exclude patterns for all projects
exclude_patterns = [
    "*.pyc", "*.pyo", "*.so", "*.dylib",  # Compiled files
    "__pycache__/*", "node_modules/*",     # Cache/deps
    ".git/*", ".svn/*",                   # Version control
    "*.log", "*.tmp", "temp/*",           # Temporary files
    "build/*", "dist/*", "target/*",      # Build outputs
]
```

### Conditional Processing

Process files differently based on characteristics:

```python
async def conditional_processing():
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Define processing rules
    rules = [
        {
            "condition": {"file_size_mb": {"<": 1}},
            "action": {"chunk_size": 500, "enable_ast": True}
        },
        {
            "condition": {"file_type": "pdf"},
            "action": {"extract_images": False, "chunk_by_pages": False}
        },
        {
            "condition": {"language": "python"},
            "action": {"chunk_by_functions": True, "include_docstrings": True}
        },
        {
            "condition": {"path": "*/tests/*"},
            "action": {"priority": "low", "chunk_size": 1000}
        }
    ]
    
    await server.index_with_rules("/path/to/project", rules=rules)
```

## Troubleshooting

### Common Issues

**Files Not Being Indexed:**
```python
# Debug file pattern matching
async def debug_patterns():
    import fnmatch
    
    patterns = ["*.py", "*.md"]
    exclude_patterns = ["*test*", "__pycache__/*"]
    
    for file_path in Path("/path/to/check").rglob("*"):
        # Check if file matches include patterns
        included = any(fnmatch.fnmatch(file_path.name, pattern) 
                      for pattern in patterns)
        
        # Check if file matches exclude patterns  
        excluded = any(fnmatch.fnmatch(str(file_path), pattern)
                      for pattern in exclude_patterns)
        
        if included and not excluded:
            print(f"✅ Would index: {file_path}")
        elif included and excluded:
            print(f"❌ Excluded: {file_path}")
        else:
            print(f"⏭️  Skipped: {file_path}")
```

**Memory Issues During Indexing:**
```python
# Monitor memory usage
import psutil

async def memory_aware_indexing():
    process = psutil.Process()
    
    def check_memory():
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > 2048:  # 2GB threshold
            print(f"⚠️ High memory usage: {memory_mb:.0f}MB")
            return False
        return True
    
    # Process files with memory checks
    files = list(Path("/path/to/project").rglob("*.py"))
    for i, file_path in enumerate(files):
        if i % 10 == 0 and not check_memory():
            # Force garbage collection
            import gc
            gc.collect()
            
        await server.index_file(file_path)
```

**Slow Processing:**
```python
# Profile indexing performance  
import time
from collections import defaultdict

async def profile_indexing():
    timings = defaultdict(list)
    
    async def timed_index(file_path):
        start = time.time()
        result = await server.index_file(file_path)
        duration = time.time() - start
        
        file_size = file_path.stat().st_size / 1024  # KB
        timings[file_path.suffix].append({
            'duration': duration,
            'size_kb': file_size,
            'chunks': result.get('chunks_created', 0)
        })
        
        return result
    
    # Profile different file types
    for pattern in ["*.py", "*.md", "*.pdf"]:
        files = list(Path("/path/to/project").rglob(pattern))[:10]  # Sample
        
        print(f"\nProfiling {pattern} files:")
        for file_path in files:
            await timed_index(file_path)
        
        # Analyze timings
        pattern_timings = timings[pattern.replace("*", "")]
        if pattern_timings:
            avg_time = sum(t['duration'] for t in pattern_timings) / len(pattern_timings)
            avg_size = sum(t['size_kb'] for t in pattern_timings) / len(pattern_timings)
            print(f"  Average: {avg_time:.2f}s per file ({avg_size:.1f} KB)")
```

## Best Practices

### Development Workflow

1. **Start Small**: Index a few representative files first
2. **Validate Chunks**: Review generated chunks for quality
3. **Tune Parameters**: Adjust chunk size and overlap
4. **Scale Gradually**: Add more files and directories
5. **Monitor Performance**: Track metrics and optimize

### Production Deployment

1. **Resource Planning**: Estimate memory and storage needs
2. **Batch Processing**: Use incremental updates for large datasets  
3. **Error Handling**: Implement robust retry mechanisms
4. **Monitoring**: Set up alerts for performance and errors
5. **Backup Strategy**: Regular index backups and recovery plans

### Quality Assurance  

1. **Content Validation**: Ensure meaningful chunk boundaries
2. **Metadata Accuracy**: Verify extracted metadata is correct
3. **Search Quality**: Test search results with known queries
4. **Performance Testing**: Validate under expected load
5. **Documentation**: Document your indexing strategy and configs

## Next Steps

Now that you understand document indexing:

1. **[Search & Retrieval](searching.md)** - Learn to find your indexed content effectively
2. **[Advanced Features](advanced-features.md)** - Explore knowledge graphs and caching
3. **[Performance Tuning](advanced-features.md#performance-optimization)** - Optimize for your specific use case
4. **[MCP Integration](integrations.md)** - Connect with Claude Desktop and other applications

Ready to make your indexed content searchable? Continue with the **[Search & Retrieval Guide](searching.md)**.