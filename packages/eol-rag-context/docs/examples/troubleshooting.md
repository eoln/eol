# Troubleshooting Examples

Comprehensive troubleshooting guide with working solutions for common issues. Each problem includes diagnostic steps, root cause analysis, and tested solutions.

## Prerequisites

Before troubleshooting, ensure you have diagnostic tools:

```bash
# Install diagnostic dependencies
pip install psutil redis-cli

# Verify EOL RAG Context installation
python -c "from eol.rag_context import EOLRAGContextServer; print('✅ Installation OK')"

# Test Redis connectivity
redis-cli ping
```

## Installation and Setup Issues

### Problem: Module Import Errors

**Symptoms:**
```
ImportError: No module named 'eol.rag_context'
ModuleNotFoundError: No module named 'sentence_transformers'
```

**Diagnosis:**
```python
import sys
import subprocess

def diagnose_installation():
    """Diagnose installation issues."""
    print("🔍 Diagnosing Installation Issues\n")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check installed packages
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True)
        installed_packages = result.stdout
        
        # Check for required packages
        required_packages = ['eol-rag-context', 'redis', 'sentence-transformers']
        
        for package in required_packages:
            if package.lower() in installed_packages.lower():
                print(f"✅ {package}: Installed")
            else:
                print(f"❌ {package}: Missing")
        
    except Exception as e:
        print(f"❌ Error checking packages: {e}")
    
    # Check import capability
    test_imports = [
        ('eol.rag_context', 'EOL RAG Context'),
        ('redis', 'Redis client'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('numpy', 'NumPy'),
        ('asyncio', 'AsyncIO')
    ]
    
    print("\n📦 Import Test:")
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"✅ {description}: OK")
        except ImportError as e:
            print(f"❌ {description}: Failed - {e}")

# Run diagnosis
diagnose_installation()
```

**Solutions:**

1. **Clean Installation:**
```bash
# Uninstall and reinstall
pip uninstall eol-rag-context -y
pip install eol-rag-context

# For development installations
pip install -e .
```

2. **Virtual Environment Setup:**
```bash
# Create clean virtual environment
python -m venv eol-env
source eol-env/bin/activate  # On Windows: eol-env\Scripts\activate
pip install --upgrade pip
pip install eol-rag-context
```

3. **Dependency Resolution:**
```bash
# Install with all dependencies
pip install eol-rag-context[all]

# Or install specific extras
pip install eol-rag-context[embedding,gpu]
```

### Problem: Redis Connection Failures

**Symptoms:**
```
redis.exceptions.ConnectionError: Error 61 connecting to localhost:6379
ConnectionRefusedError: [Errno 61] Connection refused
```

**Diagnosis:**
```python
import redis
import socket
import subprocess

async def diagnose_redis_connection():
    """Comprehensive Redis connection diagnosis."""
    print("🔍 Diagnosing Redis Connection Issues\n")
    
    # Test basic network connectivity
    def test_port(host, port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    # Check if Redis port is accessible
    redis_host = "localhost"
    redis_port = 6379
    
    print(f"Testing connection to {redis_host}:{redis_port}...")
    if test_port(redis_host, redis_port):
        print("✅ Port is accessible")
    else:
        print("❌ Port is not accessible")
        print("   Possible causes:")
        print("   - Redis not running")
        print("   - Firewall blocking connection")
        print("   - Redis bound to different interface")
    
    # Check Redis process
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'redis-server' in result.stdout:
            print("✅ Redis server process found")
        else:
            print("❌ Redis server process not found")
    except Exception:
        print("⚠️  Could not check Redis process")
    
    # Test Redis client connection
    for config in [
        {"host": "localhost", "port": 6379},
        {"host": "127.0.0.1", "port": 6379},
        {"host": "localhost", "port": 6379, "socket_connect_timeout": 5}
    ]:
        try:
            print(f"\nTesting Redis connection: {config}")
            client = redis.Redis(**config)
            response = client.ping()
            print(f"✅ Connection successful: {response}")
            
            # Test basic operations
            client.set("test_key", "test_value")
            value = client.get("test_key")
            client.delete("test_key")
            print(f"✅ Basic operations work: {value}")
            
            break
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
    
    # Check Redis configuration
    try:
        client = redis.Redis(host="localhost", port=6379)
        info = client.info()
        
        print(f"\n📊 Redis Information:")
        print(f"   Version: {info.get('redis_version', 'N/A')}")
        print(f"   Mode: {info.get('redis_mode', 'N/A')}")
        print(f"   Memory: {info.get('used_memory_human', 'N/A')}")
        print(f"   Connected clients: {info.get('connected_clients', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Could not get Redis info: {e}")

# Run diagnosis
import asyncio
asyncio.run(diagnose_redis_connection())
```

**Solutions:**

1. **Start Redis Stack:**
```bash
# Using Docker (recommended)
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

# Verify it's running
docker ps | grep redis-stack

# Check logs if issues
docker logs redis-stack
```

2. **Using Homebrew (macOS):**
```bash
# Install Redis Stack
brew install redis-stack

# Start Redis
brew services start redis-stack

# Or run manually
redis-stack-server --port 6379
```

3. **Connection Configuration:**
```python
# Connection with retry logic
import redis
import time

def create_redis_connection(max_retries=3, delay=1):
    """Create Redis connection with retry logic."""
    for attempt in range(max_retries):
        try:
            client = redis.Redis(
                host='localhost',
                port=6379,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            client.ping()
            print(f"✅ Redis connected on attempt {attempt + 1}")
            return client
            
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise
    
    return None

# Usage
redis_client = create_redis_connection()
```

## Indexing Issues

### Problem: Files Not Being Indexed

**Symptoms:**
- `indexed_files: 0` despite files being present
- Specific file types being skipped
- Permission errors

**Diagnosis:**
```python
import os
from pathlib import Path
import fnmatch

def diagnose_indexing_issues(directory_path, file_patterns=None, exclude_patterns=None):
    """Diagnose why files aren't being indexed."""
    print(f"🔍 Diagnosing Indexing Issues for: {directory_path}\n")
    
    directory = Path(directory_path)
    
    # Check directory existence and permissions
    if not directory.exists():
        print(f"❌ Directory does not exist: {directory_path}")
        return
    
    if not directory.is_dir():
        print(f"❌ Path is not a directory: {directory_path}")
        return
    
    if not os.access(directory, os.R_OK):
        print(f"❌ No read permission for directory: {directory_path}")
        return
    
    print(f"✅ Directory accessible: {directory_path}")
    
    # Default patterns
    if file_patterns is None:
        file_patterns = ["*.py", "*.md", "*.txt", "*.json"]
    
    if exclude_patterns is None:
        exclude_patterns = ["*.pyc", "__pycache__/*", ".git/*"]
    
    print(f"📋 Include patterns: {file_patterns}")
    print(f"📋 Exclude patterns: {exclude_patterns}")
    
    # Scan directory
    all_files = []
    included_files = []
    excluded_files = []
    permission_errors = []
    
    try:
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                all_files.append(file_path)
                
                # Check permissions
                try:
                    if not os.access(file_path, os.R_OK):
                        permission_errors.append(file_path)
                        continue
                except Exception:
                    permission_errors.append(file_path)
                    continue
                
                # Check include patterns
                included = any(fnmatch.fnmatch(file_path.name, pattern) for pattern in file_patterns)
                
                # Check exclude patterns
                excluded = any(fnmatch.fnmatch(str(file_path.relative_to(directory)), pattern) 
                              for pattern in exclude_patterns)
                
                if included and not excluded:
                    included_files.append(file_path)
                elif excluded:
                    excluded_files.append(file_path)
                    
    except Exception as e:
        print(f"❌ Error scanning directory: {e}")
        return
    
    # Report results
    print(f"\n📊 File Analysis:")
    print(f"   Total files found: {len(all_files)}")
    print(f"   Files to be indexed: {len(included_files)}")
    print(f"   Files excluded: {len(excluded_files)}")
    print(f"   Permission errors: {len(permission_errors)}")
    
    # Show examples
    if included_files:
        print(f"\n✅ Files that would be indexed (showing first 10):")
        for file_path in included_files[:10]:
            print(f"   📄 {file_path.relative_to(directory)}")
    
    if excluded_files:
        print(f"\n⏭️ Excluded files (showing first 10):")
        for file_path in excluded_files[:10]:
            print(f"   🚫 {file_path.relative_to(directory)}")
    
    if permission_errors:
        print(f"\n❌ Permission errors (showing first 10):")
        for file_path in permission_errors[:10]:
            print(f"   🔒 {file_path.relative_to(directory)}")
    
    # File type analysis
    file_extensions = {}
    for file_path in all_files:
        ext = file_path.suffix.lower()
        file_extensions[ext] = file_extensions.get(ext, 0) + 1
    
    print(f"\n📈 File types in directory:")
    for ext, count in sorted(file_extensions.items(), key=lambda x: x[1], reverse=True)[:15]:
        status = "✅" if any(fnmatch.fnmatch(f"*{ext}", pattern) for pattern in file_patterns) else "⏭️"
        print(f"   {status} {ext}: {count} files")

# Example usage
diagnose_indexing_issues(
    "/path/to/your/project",
    file_patterns=["*.py", "*.md", "*.json"],
    exclude_patterns=["*.pyc", "__pycache__/*", ".git/*", "node_modules/*"]
)
```

**Solutions:**

1. **Fix File Patterns:**
```python
# Correct pattern usage
correct_patterns = [
    "*.py",           # Python files
    "*.js",           # JavaScript files  
    "*.md",           # Markdown files
    "*.rst",          # reStructuredText
    "*.txt",          # Text files
    "*.json",         # JSON files
    "*.yaml", "*.yml" # YAML files
]

# Common exclude patterns
exclude_patterns = [
    "*.pyc", "*.pyo",           # Python compiled files
    "__pycache__/*",            # Python cache directories
    "node_modules/*",           # Node.js dependencies
    ".git/*", ".svn/*",         # Version control
    "build/*", "dist/*",        # Build outputs
    "*.log", "*.tmp",           # Temporary files
    ".env", ".env.*"            # Environment files
]
```

2. **Permission Issues:**
```python
import stat
import os

def fix_file_permissions(directory_path):
    """Fix common file permission issues."""
    directory = Path(directory_path)
    
    fixed_files = 0
    for file_path in directory.rglob("*"):
        try:
            if file_path.is_file() and not os.access(file_path, os.R_OK):
                # Add read permission
                current_mode = file_path.stat().st_mode
                new_mode = current_mode | stat.S_IRUSR
                file_path.chmod(new_mode)
                fixed_files += 1
                
        except Exception as e:
            print(f"Could not fix permissions for {file_path}: {e}")
    
    print(f"Fixed permissions for {fixed_files} files")

# Usage
fix_file_permissions("/path/to/your/project")
```

### Problem: Out of Memory During Indexing

**Symptoms:**
```
MemoryError: Unable to allocate array
Process killed (OOM)
```

**Diagnosis:**
```python
import psutil
import gc
from pathlib import Path

def diagnose_memory_issues():
    """Diagnose memory usage during indexing."""
    print("🔍 Diagnosing Memory Issues\n")
    
    # Current memory usage
    process = psutil.Process()
    memory_info = process.memory_info()
    
    print(f"💾 Current Memory Usage:")
    print(f"   RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
    print(f"   VMS: {memory_info.vms / 1024 / 1024:.1f} MB")
    
    # System memory
    system_memory = psutil.virtual_memory()
    print(f"   System total: {system_memory.total / 1024 / 1024 / 1024:.1f} GB")
    print(f"   System available: {system_memory.available / 1024 / 1024 / 1024:.1f} GB")
    print(f"   System usage: {system_memory.percent:.1f}%")
    
    # Check for memory leaks
    gc.collect()
    objects = gc.get_objects()
    print(f"   Python objects: {len(objects)}")
    
    # Estimate indexing memory requirements
    def estimate_memory_needs(directory_path, chunk_size=1000):
        directory = Path(directory_path)
        total_size = 0
        file_count = 0
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                    file_count += 1
                except Exception:
                    continue
        
        # Rough estimation: 
        # - Text takes ~2x size in memory (Unicode)
        # - Embeddings: ~1.5KB per chunk (384 dim float32)
        # - Processing overhead: ~3x
        
        estimated_chunks = total_size / chunk_size
        text_memory = total_size * 2  # Text in memory
        embedding_memory = estimated_chunks * 1536  # Embeddings
        processing_overhead = (text_memory + embedding_memory) * 3
        
        total_estimated = text_memory + embedding_memory + processing_overhead
        
        print(f"\n📊 Memory Requirements Estimate:")
        print(f"   Directory size: {total_size / 1024 / 1024:.1f} MB")
        print(f"   Files: {file_count}")
        print(f"   Estimated chunks: {estimated_chunks:.0f}")
        print(f"   Estimated memory needed: {total_estimated / 1024 / 1024:.1f} MB")
        
        return total_estimated
    
    return estimate_memory_needs

# Usage
memory_estimator = diagnose_memory_issues()
memory_estimator("/path/to/your/project")
```

**Solutions:**

1. **Memory-Efficient Configuration:**
```python
# Memory-optimized configuration
memory_config = {
    "embedding": {
        "batch_size": 8,            # Smaller batches
        "cache_embeddings": False,  # Don't cache in memory
        "streaming_mode": True      # Process in streams
    },
    "indexing": {
        "chunk_buffer_size": 50,    # Smaller buffer
        "gc_interval": 10,          # Frequent garbage collection
        "memory_limit_mb": 2048,    # Set memory limit
        "parallel_workers": 1       # Single worker to save memory
    },
    "redis": {
        "pipeline_size": 10         # Smaller Redis pipelines
    }
}

# Apply configuration
server = EOLRAGContextServer(config=memory_config)
```

2. **Batch Processing:**
```python
import asyncio
import gc
from pathlib import Path

async def memory_conscious_indexing(server, directory_path, batch_size=100):
    """Index files in batches to manage memory usage."""
    
    directory = Path(directory_path)
    all_files = list(directory.rglob("*.py")) + list(directory.rglob("*.md"))
    
    print(f"📁 Found {len(all_files)} files to index")
    
    # Process in batches
    for i in range(0, len(all_files), batch_size):
        batch = all_files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}: {len(batch)} files")
        
        # Index batch
        for file_path in batch:
            try:
                await server.index_file(str(file_path))
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")
        
        # Force garbage collection
        gc.collect()
        
        # Check memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage: {memory_mb:.1f} MB")
        
        # Optional: pause between batches
        await asyncio.sleep(1)
    
    print("✅ Batch indexing completed")

# Usage
async def run_batch_indexing():
    server = EOLRAGContextServer(config=memory_config)
    await server.initialize()
    
    try:
        await memory_conscious_indexing(server, "/path/to/project", batch_size=50)
    finally:
        await server.close()

asyncio.run(run_batch_indexing())
```

## Search Issues

### Problem: Poor Search Results

**Symptoms:**
- Low similarity scores
- Irrelevant results
- Missing expected content

**Diagnosis:**
```python
import asyncio
from eol.rag_context import EOLRAGContextServer

async def diagnose_search_quality():
    """Diagnose search quality issues."""
    print("🔍 Diagnosing Search Quality Issues\n")
    
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Test queries with analysis
    test_cases = [
        {
            'query': 'user authentication',
            'expected_terms': ['user', 'auth', 'login', 'password', 'token'],
            'expected_file_types': ['.py', '.md']
        },
        {
            'query': 'database configuration',
            'expected_terms': ['database', 'config', 'connection', 'settings'],
            'expected_file_types': ['.yaml', '.json', '.py']
        },
        {
            'query': 'how to install dependencies',
            'expected_terms': ['install', 'dependencies', 'requirements', 'pip'],
            'expected_file_types': ['.md', '.txt']
        }
    ]
    
    for test_case in test_cases:
        query = test_case['query']
        print(f"🧪 Testing query: '{query}'")
        
        # Search with different parameters
        for threshold in [0.5, 0.7, 0.8]:
            result = await server.search_context({
                'query': query,
                'max_results': 10,
                'similarity_threshold': threshold
            }, None)
            
            results = result['results']
            print(f"   Threshold {threshold}: {len(results)} results")
            
            if results:
                avg_similarity = sum(r['similarity'] for r in results) / len(results)
                print(f"   Average similarity: {avg_similarity:.3f}")
                
                # Analyze result quality
                file_types = set(Path(r['source_path']).suffix for r in results)
                print(f"   File types found: {file_types}")
                
                # Check for expected terms
                found_terms = set()
                for result in results:
                    content_lower = result['content'].lower()
                    for term in test_case['expected_terms']:
                        if term.lower() in content_lower:
                            found_terms.add(term)
                
                print(f"   Expected terms found: {found_terms}")
                
                # Show best result
                if results:
                    best = results[0]
                    print(f"   Best result: {Path(best['source_path']).name} ({best['similarity']:.3f})")
            
            print()
    
    await server.close()

# Run diagnosis
asyncio.run(diagnose_search_quality())
```

**Solutions:**

1. **Query Optimization:**
```python
def optimize_query(original_query):
    """Optimize queries for better search results."""
    
    query_improvements = {
        # Add context for ambiguous terms
        'auth': 'user authentication system',
        'config': 'configuration settings',
        'setup': 'installation and setup',
        
        # Expand abbreviations
        'db': 'database',
        'api': 'API endpoint',
        'docs': 'documentation',
        
        # Add specific terms for better matching
        'error': 'error handling and debugging',
        'test': 'testing and unit tests'
    }
    
    # Apply improvements
    optimized = original_query.lower()
    for short_term, expansion in query_improvements.items():
        if short_term in optimized:
            optimized = optimized.replace(short_term, expansion)
    
    return optimized

# Usage examples
print(optimize_query("auth setup"))        # → "user authentication system installation and setup"
print(optimize_query("db config"))         # → "database configuration settings"
```

2. **Search Parameter Tuning:**
```python
async def find_optimal_search_params(server, query, ground_truth_files=None):
    """Find optimal search parameters for a query."""
    
    print(f"🎯 Tuning parameters for: '{query}'\n")
    
    # Test different parameter combinations
    param_combinations = [
        {'similarity_threshold': 0.5, 'max_results': 10},
        {'similarity_threshold': 0.6, 'max_results': 10},
        {'similarity_threshold': 0.7, 'max_results': 10},
        {'similarity_threshold': 0.8, 'max_results': 10},
        {'similarity_threshold': 0.7, 'max_results': 5},
        {'similarity_threshold': 0.7, 'max_results': 15},
    ]
    
    best_params = None
    best_score = 0
    
    for params in param_combinations:
        result = await server.search_context({
            'query': query,
            **params
        }, None)
        
        results = result['results']
        
        if results:
            avg_similarity = sum(r['similarity'] for r in results) / len(results)
            
            # Calculate quality score
            quality_score = avg_similarity * len(results) / 10  # Normalize by result count
            
            # If we have ground truth, check precision
            if ground_truth_files:
                found_files = set(r['source_path'] for r in results)
                precision = len(found_files.intersection(ground_truth_files)) / len(found_files)
                quality_score *= precision
            
            print(f"Params {params}: Score {quality_score:.3f} (avg sim: {avg_similarity:.3f}, count: {len(results)})")
            
            if quality_score > best_score:
                best_score = quality_score
                best_params = params
        else:
            print(f"Params {params}: No results")
    
    print(f"\n🏆 Best parameters: {best_params} (score: {best_score:.3f})")
    return best_params

# Usage
# best_params = await find_optimal_search_params(server, "user authentication")
```

### Problem: Slow Search Performance

**Symptoms:**
- Search taking > 5 seconds
- High CPU usage during search
- Memory spikes

**Diagnosis:**
```python
import time
import cProfile
import pstats
from io import StringIO

async def diagnose_search_performance():
    """Diagnose search performance issues."""
    print("🔍 Diagnosing Search Performance\n")
    
    server = EOLRAGContextServer()
    await server.initialize()
    
    # Test queries with timing
    test_queries = [
        "simple query",
        "more complex query with multiple terms",
        "very long query with many specific technical terms and detailed context",
        "authentication system implementation",
        "database configuration and setup"
    ]
    
    performance_data = []
    
    for query in test_queries:
        print(f"⏱️  Testing: '{query[:30]}...' ")
        
        # Time the search
        start_time = time.time()
        
        result = await server.search_context({
            'query': query,
            'max_results': 10
        }, None)
        
        end_time = time.time()
        search_time = (end_time - start_time) * 1000  # Convert to ms
        
        cache_hit = result.get('cache_hit', False)
        result_count = len(result['results'])
        
        performance_data.append({
            'query': query,
            'time_ms': search_time,
            'cache_hit': cache_hit,
            'results': result_count
        })
        
        status = "CACHED" if cache_hit else "NEW"
        print(f"   {search_time:.1f}ms ({result_count} results) [{status}]")
    
    # Analyze performance patterns
    cache_hits = [d for d in performance_data if d['cache_hit']]
    cache_misses = [d for d in performance_data if not d['cache_hit']]
    
    if cache_hits:
        avg_hit_time = sum(d['time_ms'] for d in cache_hits) / len(cache_hits)
        print(f"\n📊 Cache hits average: {avg_hit_time:.1f}ms")
    
    if cache_misses:
        avg_miss_time = sum(d['time_ms'] for d in cache_misses) / len(cache_misses)
        print(f"📊 Cache misses average: {avg_miss_time:.1f}ms")
    
    # Profile a slow search
    if cache_misses and max(d['time_ms'] for d in cache_misses) > 1000:
        print(f"\n🔬 Profiling slow search...")
        
        profiler = cProfile.Profile()
        
        slow_query = max(cache_misses, key=lambda x: x['time_ms'])['query']
        
        profiler.enable()
        
        await server.search_context({
            'query': slow_query,
            'max_results': 10
        }, None)
        
        profiler.disable()
        
        # Show profile results
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        
        print("Top time-consuming functions:")
        print(s.getvalue())
    
    await server.close()

# Run diagnosis
asyncio.run(diagnose_search_performance())
```

**Solutions:**

1. **Enable and Optimize Caching:**
```python
# Optimized cache configuration
cache_config = {
    "caching": {
        "enabled": True,
        "ttl_seconds": 3600,           # 1 hour cache
        "similarity_threshold": 0.95,  # Cache very similar queries
        "max_cache_size": 5000,        # Large cache
        "target_hit_rate": 0.31,       # Optimal hit rate
        "adaptive_threshold": True,    # Auto-adjust threshold
        "quality_scoring": True        # Score cached responses
    }
}

server = EOLRAGContextServer(config=cache_config)
```

2. **Search Optimization:**
```python
async def optimized_search(server, query, **kwargs):
    """Optimized search with performance monitoring."""
    
    # Set performance-optimized defaults
    search_params = {
        'max_results': 5,              # Limit results
        'similarity_threshold': 0.75,  # Higher threshold
        'early_termination': True,     # Stop when enough results found
        **kwargs
    }
    
    start_time = time.time()
    
    try:
        result = await server.search_context({
            'query': query,
            **search_params
        }, None)
        
        search_time = (time.time() - start_time) * 1000
        
        # Log performance
        print(f"Search completed in {search_time:.1f}ms")
        print(f"Cache hit: {result.get('cache_hit', False)}")
        print(f"Results: {len(result['results'])}")
        
        return result
        
    except Exception as e:
        print(f"Search failed: {e}")
        raise

# Usage
# result = await optimized_search(server, "user authentication")
```

## Production Issues

### Problem: Memory Leaks in Production

**Symptoms:**
- Memory usage continuously growing
- Eventually runs out of memory
- Performance degrades over time

**Diagnosis and Solution:**
```python
import gc
import weakref
import psutil
import asyncio
from collections import defaultdict

class MemoryMonitor:
    """Monitor and track memory usage patterns."""
    
    def __init__(self):
        self.baseline_memory = None
        self.measurements = []
        self.object_counts = defaultdict(int)
    
    def start_monitoring(self):
        """Start memory monitoring."""
        process = psutil.Process()
        self.baseline_memory = process.memory_info().rss / 1024 / 1024
        print(f"📊 Baseline memory: {self.baseline_memory:.1f} MB")
    
    def measure(self, label=""):
        """Take a memory measurement."""
        # Force garbage collection
        gc.collect()
        
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024
        
        # Count objects by type
        objects = gc.get_objects()
        object_types = defaultdict(int)
        for obj in objects:
            obj_type = type(obj).__name__
            object_types[obj_type] += 1
        
        # Record measurement
        measurement = {
            'label': label,
            'memory_mb': current_memory,
            'memory_delta': current_memory - self.baseline_memory,
            'object_count': len(objects),
            'top_objects': dict(sorted(object_types.items(), key=lambda x: x[1], reverse=True)[:10])
        }
        
        self.measurements.append(measurement)
        
        print(f"📊 {label}: {current_memory:.1f} MB (+{measurement['memory_delta']:.1f} MB)")
        
        return measurement
    
    def detect_leaks(self):
        """Detect potential memory leaks."""
        if len(self.measurements) < 2:
            return []
        
        # Look for consistently growing object types
        leaks = []
        
        first_objects = self.measurements[0]['top_objects']
        last_objects = self.measurements[-1]['top_objects']
        
        for obj_type in first_objects:
            if obj_type in last_objects:
                growth = last_objects[obj_type] - first_objects[obj_type]
                if growth > 100:  # Significant growth
                    leaks.append({
                        'type': obj_type,
                        'growth': growth,
                        'initial': first_objects[obj_type],
                        'final': last_objects[obj_type]
                    })
        
        return leaks
    
    def report(self):
        """Generate memory usage report."""
        print(f"\n📊 Memory Usage Report:")
        print(f"   Measurements: {len(self.measurements)}")
        
        if self.measurements:
            max_memory = max(m['memory_mb'] for m in self.measurements)
            total_growth = self.measurements[-1]['memory_delta']
            
            print(f"   Peak memory: {max_memory:.1f} MB")
            print(f"   Total growth: {total_growth:.1f} MB")
        
        # Check for leaks
        leaks = self.detect_leaks()
        if leaks:
            print(f"\n⚠️  Potential memory leaks detected:")
            for leak in leaks:
                print(f"   {leak['type']}: {leak['initial']} → {leak['final']} (+{leak['growth']})")
        else:
            print(f"\n✅ No significant memory leaks detected")

async def test_for_memory_leaks():
    """Test EOL RAG Context for memory leaks."""
    
    monitor = MemoryMonitor()
    monitor.start_monitoring()
    
    server = EOLRAGContextServer()
    await server.initialize()
    
    monitor.measure("After initialization")
    
    # Simulate typical usage patterns
    queries = [
        "user authentication system",
        "database configuration",
        "API endpoint documentation",
        "error handling patterns",
        "testing and validation"
    ]
    
    # Run multiple cycles to detect leaks
    for cycle in range(5):
        print(f"\n🔄 Cycle {cycle + 1}")
        
        # Index some content (simulate new content)
        monitor.measure(f"Cycle {cycle + 1} - Before indexing")
        
        # Simulate search activity
        for query in queries:
            try:
                result = await server.search_context({
                    'query': query,
                    'max_results': 10
                }, None)
            except Exception as e:
                print(f"Search error: {e}")
        
        monitor.measure(f"Cycle {cycle + 1} - After searches")
        
        # Force cleanup
        gc.collect()
        
        monitor.measure(f"Cycle {cycle + 1} - After cleanup")
    
    await server.close()
    monitor.measure("After server close")
    
    # Generate report
    monitor.report()

# Run memory leak test
asyncio.run(test_for_memory_leaks())
```

### Problem: High CPU Usage

**Diagnosis and Solution:**
```python
import psutil
import asyncio
import cProfile
import threading
import time

class CPUMonitor:
    """Monitor CPU usage patterns."""
    
    def __init__(self):
        self.monitoring = False
        self.measurements = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start CPU monitoring in background thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("📊 CPU monitoring started")
    
    def stop_monitoring(self):
        """Stop CPU monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        print("📊 CPU monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                process = psutil.Process()
                process_cpu = process.cpu_percent()
                
                self.measurements.append({
                    'timestamp': time.time(),
                    'system_cpu': cpu_percent,
                    'process_cpu': process_cpu
                })
                
                if process_cpu > 80:  # High CPU usage
                    print(f"⚠️  High CPU usage: {process_cpu:.1f}%")
                
            except Exception as e:
                print(f"CPU monitoring error: {e}")
            
            time.sleep(1)
    
    def report(self):
        """Generate CPU usage report."""
        if not self.measurements:
            print("No CPU measurements recorded")
            return
        
        system_cpu_avg = sum(m['system_cpu'] for m in self.measurements) / len(self.measurements)
        process_cpu_avg = sum(m['process_cpu'] for m in self.measurements) / len(self.measurements)
        
        system_cpu_max = max(m['system_cpu'] for m in self.measurements)
        process_cpu_max = max(m['process_cpu'] for m in self.measurements)
        
        print(f"\n📊 CPU Usage Report:")
        print(f"   System CPU - Average: {system_cpu_avg:.1f}%, Peak: {system_cpu_max:.1f}%")
        print(f"   Process CPU - Average: {process_cpu_avg:.1f}%, Peak: {process_cpu_max:.1f}%")
        
        # Identify high usage periods
        high_usage = [m for m in self.measurements if m['process_cpu'] > 50]
        if high_usage:
            print(f"   High usage periods: {len(high_usage)} measurements")

async def diagnose_cpu_usage():
    """Diagnose CPU usage issues."""
    
    cpu_monitor = CPUMonitor()
    cpu_monitor.start_monitoring()
    
    server = EOLRAGContextServer()
    await server.initialize()
    
    try:
        # Test different operations
        operations = [
            ("Indexing", lambda: server.index_directory("/path/to/small/project")),
            ("Searching", lambda: server.search_context({'query': 'test query'}, None)),
            ("Bulk search", lambda: asyncio.gather(*[
                server.search_context({'query': f'query {i}'}, None) 
                for i in range(10)
            ]))
        ]
        
        for op_name, operation in operations:
            print(f"\n🔄 Testing {op_name}...")
            
            start_time = time.time()
            
            try:
                await operation()
            except Exception as e:
                print(f"Operation failed: {e}")
            
            duration = time.time() - start_time
            print(f"   Completed in {duration:.1f}s")
            
            # Brief pause between operations
            await asyncio.sleep(2)
    
    finally:
        cpu_monitor.stop_monitoring()
        await server.close()
        
        cpu_monitor.report()

# Run CPU diagnosis
asyncio.run(diagnose_cpu_usage())
```

## Best Practices for Troubleshooting

### Systematic Troubleshooting Approach

1. **Gather Information:**
```python
def gather_system_info():
    """Gather comprehensive system information for troubleshooting."""
    
    info = {
        'system': {
            'platform': platform.platform(),
            'python_version': sys.version,
            'architecture': platform.architecture()
        },
        'memory': {
            'total_gb': psutil.virtual_memory().total / (1024**3),
            'available_gb': psutil.virtual_memory().available / (1024**3),
            'percent_used': psutil.virtual_memory().percent
        },
        'cpu': {
            'count': psutil.cpu_count(),
            'current_usage': psutil.cpu_percent(interval=1)
        },
        'disk': {
            'free_gb': psutil.disk_usage('/').free / (1024**3),
            'total_gb': psutil.disk_usage('/').total / (1024**3)
        }
    }
    
    print("🔍 System Information:")
    for category, data in info.items():
        print(f"   {category.title()}:")
        for key, value in data.items():
            print(f"      {key}: {value}")
    
    return info
```

2. **Enable Debug Logging:**
```python
import logging

def setup_debug_logging():
    """Setup comprehensive debug logging."""
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('eol_debug.log'),
            logging.StreamHandler()
        ]
    )
    
    # Enable specific logger categories
    loggers = [
        'eol.rag_context',
        'redis',
        'sentence_transformers'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
    
    print("📝 Debug logging enabled")
```

3. **Create Minimal Reproduction:**
```python
async def minimal_reproduction_test():
    """Create minimal test case to reproduce issues."""
    
    print("🧪 Minimal Reproduction Test")
    
    # Simplest possible setup
    server = EOLRAGContextServer(config={
        "redis": {"url": "redis://localhost:6379"},
        "embedding": {"provider": "sentence_transformers"}
    })
    
    try:
        await server.initialize()
        
        # Test basic functionality
        result = await server.search_context({
            'query': 'test'
        }, None)
        
        print(f"✅ Basic search works: {len(result['results'])} results")
        
    except Exception as e:
        print(f"❌ Basic functionality failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if server:
            await server.close()

# Run minimal test
asyncio.run(minimal_reproduction_test())
```

### Getting Help

When troubleshooting doesn't resolve the issue:

1. **Collect Diagnostic Information:**
   - System information (OS, Python version, memory, CPU)
   - Error messages and stack traces
   - Configuration used
   - Steps to reproduce

2. **Create Issue Report:**
   - Use the diagnostic scripts from this guide
   - Include logs with debug logging enabled
   - Provide minimal reproduction case
   - Specify expected vs actual behavior

3. **Community Resources:**
   - Check existing issues on GitHub
   - Search documentation for similar problems
   - Join community discussions

### Prevention Strategies

1. **Monitoring Setup:**
```python
# Add to your production setup
async def setup_monitoring():
    """Setup comprehensive monitoring."""
    
    config = {
        "monitoring": {
            "enabled": True,
            "memory_threshold_mb": 2048,
            "cpu_threshold_percent": 80,
            "response_time_threshold_ms": 1000,
            "error_rate_threshold": 0.05
        }
    }
    
    server = EOLRAGContextServer(config=config)
    # Additional monitoring setup...
```

2. **Health Checks:**
```python
async def health_check():
    """Comprehensive health check."""
    
    checks = {
        'redis': await check_redis_health(),
        'memory': check_memory_usage(),
        'embeddings': await test_embeddings(),
        'search': await test_search()
    }
    
    all_healthy = all(check['healthy'] for check in checks.values())
    
    return {
        'status': 'healthy' if all_healthy else 'unhealthy',
        'checks': checks
    }
```

3. **Regular Maintenance:**
```python
async def maintenance_tasks():
    """Regular maintenance tasks."""
    
    # Clear old cache entries
    await server.clear_expired_cache()
    
    # Optimize index
    await server.optimize_index()
    
    # Check for memory leaks
    gc.collect()
    
    # Verify system health
    await health_check()
```

Use these troubleshooting examples as a starting point for diagnosing and resolving issues with EOL RAG Context. Each example provides both diagnostic tools and practical solutions.