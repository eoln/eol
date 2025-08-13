# Redis Vector Operations

## Index Design

### Creating Vector Index
```python
from redis.commands.search.field import VectorField, TextField, TagField

schema = [
    TextField("content"),
    TagField("source"),
    TagField("file_type"),
    VectorField(
        "embedding",
        algorithm="HNSW",  # or "FLAT" for small datasets
        attributes={
            "TYPE": "FLOAT32",
            "DIM": 384,  # dimension of embeddings
            "DISTANCE_METRIC": "COSINE",
            "M": 16,  # HNSW parameter
            "EF_CONSTRUCTION": 200
        }
    )
]
```

## Vector Search Patterns

### Basic Vector Search
```python
async def vector_search(query_embedding: List[float], k: int = 5):
    query = (
        Query(f"*=>[KNN {k} @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("content", "source", "score")
        .dialect(2)
    )
    
    params = {"vec": np.array(query_embedding).tobytes()}
    return await redis_client.ft().search(query, query_params=params)
```

### Hybrid Search (Vector + Filters)
```python
async def hybrid_search(
    query_embedding: List[float],
    filters: Dict[str, Any],
    k: int = 5
):
    # Build filter string
    filter_str = " ".join([
        f"@{field}:{{{value}}}" 
        for field, value in filters.items()
    ])
    
    query = (
        Query(f"({filter_str})=>[KNN {k} @embedding $vec AS score]")
        .sort_by("score")
        .return_fields("content", "source", "score")
    )
    
    return await redis_client.ft().search(query, query_params=params)
```

## Performance Optimization

### Batch Operations
```python
async def batch_index_vectors(documents: List[Document]):
    pipe = redis_client.pipeline()
    
    for doc in documents:
        key = f"doc:{doc.id}"
        pipe.hset(key, mapping={
            "content": doc.content,
            "embedding": doc.embedding.tobytes(),
            "source": doc.source
        })
    
    await pipe.execute()
```

### Index Configuration Tips
- HNSW: Better for large datasets (>10k vectors)
- FLAT: Better for small datasets (<10k vectors)
- Adjust M and EF_CONSTRUCTION for quality/speed tradeoff
- Monitor index size and query performance