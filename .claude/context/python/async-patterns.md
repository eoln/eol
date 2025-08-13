# Async/Await Best Practices for RAG

## Basic Async Patterns

### Async Function Structure
```python
async def process_document(doc: Document) -> ProcessedDocument:
    """Process document asynchronously"""
    # Async I/O operations
    content = await read_file(doc.path)
    chunks = await chunk_content(content)
    embeddings = await generate_embeddings(chunks)
    
    # CPU-bound work in executor
    processed = await asyncio.get_event_loop().run_in_executor(
        None, heavy_processing, chunks
    )
    
    return ProcessedDocument(chunks, embeddings)
```

## Concurrent Operations

### Parallel Processing
```python
async def index_documents(docs: List[Document]) -> List[Result]:
    """Process multiple documents concurrently"""
    # Create tasks for parallel execution
    tasks = [process_document(doc) for doc in docs]
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle any errors
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Failed to process {docs[i]}: {result}")
    
    return [r for r in results if not isinstance(r, Exception)]
```

### Rate Limiting
```python
class RateLimiter:
    def __init__(self, rate: int, per: float):
        self.rate = rate
        self.per = per
        self.semaphore = asyncio.Semaphore(rate)
        
    async def __aenter__(self):
        await self.semaphore.acquire()
        return self
        
    async def __aexit__(self, *args):
        await asyncio.sleep(self.per / self.rate)
        self.semaphore.release()

# Usage
limiter = RateLimiter(rate=10, per=1.0)  # 10 requests per second

async def rate_limited_operation():
    async with limiter:
        return await api_call()
```

## Context Managers

### Async Resource Management
```python
class AsyncRedisConnection:
    async def __aenter__(self):
        self.client = await create_redis_client()
        return self.client
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()

# Usage
async with AsyncRedisConnection() as redis:
    await redis.set("key", "value")
```

## Error Handling in Async

### Retry with Backoff
```python
async def retry_async(
    func, 
    max_attempts: int = 3,
    backoff_factor: float = 2.0
):
    """Retry async function with exponential backoff"""
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            wait_time = backoff_factor ** attempt
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s")
            await asyncio.sleep(wait_time)
```

## Queue Processing

### Async Queue Pattern
```python
async def worker(queue: asyncio.Queue, worker_id: int):
    """Process items from queue"""
    while True:
        item = await queue.get()
        if item is None:  # Shutdown signal
            break
        try:
            await process_item(item)
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
        finally:
            queue.task_done()

async def process_with_workers(items: List[Any], num_workers: int = 5):
    """Process items using worker pool"""
    queue = asyncio.Queue()
    
    # Add items to queue
    for item in items:
        await queue.put(item)
    
    # Create workers
    workers = [
        asyncio.create_task(worker(queue, i))
        for i in range(num_workers)
    ]
    
    # Wait for all items to be processed
    await queue.join()
    
    # Shutdown workers
    for _ in range(num_workers):
        await queue.put(None)
    
    await asyncio.gather(*workers)
```

## Best Practices
1. Never use blocking I/O in async functions
2. Use `asyncio.gather()` for parallel operations
3. Implement proper cancellation handling
4. Use async context managers for resources
5. Avoid mixing sync and async code
6. Use `asyncio.create_task()` for fire-and-forget
7. Implement timeouts for long operations