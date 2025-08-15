# Vector Search Patterns

## Query Optimization

### Hybrid Search Strategy

```python
async def hybrid_search(query: str, k: int = 5) -> List[Document]:
    """Combine vector and keyword search"""
    # Vector search for semantic similarity
    vector_results = await vector_search(query, k=k*2)

    # Keyword search for exact matches
    keyword_results = await keyword_search(query, k=k*2)

    # Merge and re-rank results
    return rerank_results(vector_results, keyword_results, k=k)
```

### Query Expansion

- Use synonyms for broader coverage
- Include related concepts
- Consider typo tolerance
- Expand abbreviations

## Result Re-ranking

1. Initial retrieval (cast wide net)
2. Score by multiple factors:
   - Semantic similarity
   - Keyword matches
   - Metadata relevance
   - Recency (if applicable)
3. Re-rank by combined score
4. Return top-k results

## Performance Optimization

- Pre-compute embeddings for static content
- Use appropriate index types (HNSW, IVF)
- Optimize dimension count (384-768 typical)
- Implement query caching
- Batch embedding generation

## Filtering Strategies

```python
# Metadata-based filtering
filters = {
    "file_type": ["py", "md"],
    "updated_after": "2024-01-01",
    "source": "packages/eol-rag-context"
}
```
