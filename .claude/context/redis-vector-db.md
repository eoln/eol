# Redis v8 Vector Database & Context Storage

## Overview

Redis v8 provides a comprehensive solution for vector storage, similarity search, and real-time context management essential for AI/LLM applications.

## Core Capabilities

### Vector Search Features

- **HNSW Index**: Hierarchical Navigable Small World index for fast k-nearest neighbor (KNN) lookups
- **Distance Metrics**: Cosine similarity, Euclidean distance, and Inner Product
- **Configurable Dimensions**: Support for various embedding sizes (e.g., DIM=512, DIM=1536)
- **Hybrid Search**: Combine vector similarity with traditional filters

### RediSearch Integration

- Create VECTOR HNSW indexes on hash fields
- Full-text search alongside vector search
- Tag, numeric, and geo filtering capabilities
- Aggregation and faceted search

## Python SDK - RedisVL

### Installation

```bash
pip install redis-vl
```

### Key Components

#### 1. Embedding Management

```python
from redisvl import EmbeddingsCache

# Initialize cache
cache = EmbeddingsCache(
    redis_client=redis_client,
    prefix="embeddings:",
    ttl=3600  # 1 hour TTL
)

# Store embeddings with metadata
cache.store(
    text="Sample text",
    embedding=vector,
    metadata={"source": "document.pdf", "page": 1}
)
```

#### 2. Vector Index Creation

```python
from redisvl import SearchIndex

# Define schema
schema = {
    "index": {
        "name": "context_index",
        "prefix": "doc:",
        "storage_type": "hash"
    },
    "fields": [
        {"name": "content", "type": "text"},
        {"name": "embedding", "type": "vector",
         "attrs": {"dims": 1536, "algorithm": "hnsw",
                   "distance_metric": "cosine"}}
    ]
}

# Create index
index = SearchIndex.from_dict(schema)
index.create(overwrite=True)
```

#### 3. Session/Context Management

```python
from redisvl import MessageHistory

# Manage conversation context
history = MessageHistory(
    redis_client=redis_client,
    session_id="user_123",
    ttl=1800  # 30 minutes
)

# Store messages
history.add_message(role="user", content="Hello")
history.add_message(role="assistant", content="Hi!")

# Retrieve context with vector search
relevant_context = history.get_relevant_messages(
    query_embedding=embedding,
    top_k=5
)
```

### Embedding Provider Support

- OpenAI
- Cohere
- HuggingFace
- VertexAI
- Custom providers via interface

## Real-Time Context Management

### LLM Session Management

- Store conversation history between LLM and users
- Fetch recent and relevant portions using vector similarity
- Manage access patterns: recency-based or relevancy-based
- TTL-based automatic cleanup

### Semantic Caching

```python
# Cache LLM responses
cache_key = f"llm:response:{hash(prompt)}"
cached = redis_client.get(cache_key)

if not cached:
    response = llm.generate(prompt)
    redis_client.setex(
        cache_key,
        ttl=3600,
        value=response
    )
```

### Performance Benefits

- Reduce computational costs via caching
- Improve response times (sub-millisecond latency)
- Scale horizontally with Redis Cluster
- Handle real-time data updates

## Integration Patterns

### RAG Implementation

```python
# 1. Index documents
for doc in documents:
    embedding = embed_model.encode(doc.content)
    redis_client.hset(
        f"doc:{doc.id}",
        mapping={
            "content": doc.content,
            "embedding": embedding.tobytes(),
            "metadata": json.dumps(doc.metadata)
        }
    )

# 2. Query with vector similarity
query_embedding = embed_model.encode(query)
results = index.search(
    query=query_embedding,
    k=10,
    return_fields=["content", "metadata"]
)

# 3. Augment LLM prompt
context = "\n".join([r.content for r in results])
prompt = f"Context: {context}\n\nQuestion: {query}"
```

### Multi-Modal Search

- Store text, image, and audio embeddings
- Cross-modal retrieval capabilities
- CLIP model integration for image-text search

## Redis Protocol Commands

### Vector Operations

```redis
# Create vector index
FT.CREATE idx:context
    ON HASH PREFIX 1 doc:
    SCHEMA
        content TEXT
        embedding VECTOR HNSW 6
            TYPE FLOAT32
            DIM 1536
            DISTANCE_METRIC COSINE

# Add document with vector
HSET doc:1
    content "Sample text content"
    embedding <binary_vector_data>

# Vector similarity search
FT.SEARCH idx:context
    "*=>[KNN 10 @embedding $vec AS score]"
    PARAMS 2 vec <query_vector>
    RETURN 2 content score
```

### Session Management

```redis
# Store session data
HSET session:user123
    last_activity 1704067200
    context "Previous conversation..."

# Set expiration
EXPIRE session:user123 1800

# Retrieve with pattern
SCAN 0 MATCH session:* COUNT 100
```

## Best Practices

### Index Design

1. Choose appropriate vector dimensions (balance accuracy vs. performance)
2. Use HNSW for large datasets (>1M vectors)
3. Implement hybrid search for better relevance
4. Monitor index memory usage

### Performance Optimization

1. Batch vector insertions
2. Use connection pooling
3. Implement result caching
4. Configure appropriate Redis persistence

### Scaling Strategies

1. Redis Cluster for horizontal scaling
2. Read replicas for query distribution
3. Separate indexes for different data types
4. Memory optimization with compression

## EOL Framework Integration

### Proposed Architecture

1. **Primary Storage**: Redis v8 as central vector store
2. **Context Layer**: RedisVL for embedding management
3. **Caching**: Semantic cache for LLM responses
4. **Session Management**: Conversation history tracking
5. **Search**: Hybrid vector + metadata filtering
6. **Real-time Updates**: Pub/Sub for context changes

### Dependencies

```python
# requirements.txt
redis>=5.0.0
redis-vl>=0.2.0
numpy>=1.24.0
```

### Connection Configuration

```python
# Redis connection for EOL
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "decode_responses": False,  # For binary vectors
    "connection_pool_kwargs": {
        "max_connections": 50
    }
}
```
