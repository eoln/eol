# RAG Context Rules

## Performance Targets
- Document indexing: >10 docs/sec
- Vector search: <100ms latency
- Cache hit rate: >31%
- Embedding generation: <50ms per chunk
- Context window usage: <80% normally

## Always Consider
- Chunking strategy for content type
- Embedding model consistency
- Metadata design
- Hierarchical indexing approach
- Semantic similarity thresholds

## Chunking Strategies
- **Code**: AST-based chunking
- **Text**: Semantic paragraph chunking
- **Markdown**: Section-based chunking
- **JSON**: Structure-preserving chunks

## Vector Search Optimization
- Use hybrid search when appropriate
- Implement result re-ranking
- Consider query expansion
- Cache frequently accessed results

## Semantic Caching
- Target 31% hit rate minimum
- Use semantic similarity for cache keys
- Implement TTL-based expiration
- Monitor cache performance metrics