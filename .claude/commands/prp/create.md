# prp-create - Generate Product Requirements Prompt

Creates comprehensive Product Requirements Prompts (PRPs) using context engineering methodology for any software feature development. Analyzes existing codebase patterns and generates implementation blueprints with curated intelligence.

## Command Overview

**Purpose**: Generate implementation blueprints using context engineering principles
**Target**: Any software feature requiring systematic development approach
**Output**: Comprehensive PRP with curated codebase intelligence saved to `.claude/plans/draft/prp-[feature-name].md`

## Usage

```bash
/prp:create [feature-description]
```

### Examples
```bash
# Any software feature development
/prp:create "user authentication system with JWT tokens"
/prp:create "real-time notification service with WebSockets"
/prp:create "hierarchical document indexing with concept extraction"
/prp:create "API rate limiting with Redis backend"
/prp:create "file upload service with virus scanning"
```

## PRP Generation Process

### 1. Codebase Analysis Phase
- **Technology Scanning**: Identify relevant frameworks, libraries, and patterns
- **Pattern Discovery**: Extract existing code conventions, structures, and best practices
- **Dependency Mapping**: Analyze package dependencies and integration points
- **Test Pattern Review**: Understand testing frameworks and validation strategies

### 2. Context Engineering Phase
- **Domain Knowledge**: Extract domain-specific patterns and conventions
- **Technical Constraints**: Identify dependencies, performance requirements, and limitations
- **Integration Context**: Understand system boundaries and external service connections
- **Quality Standards**: Establish testing, validation, and performance baselines

### 3. Implementation Blueprint Creation
- **Architecture Design**: Component structure following existing patterns
- **Implementation Strategy**: Step-by-step development approach with validation gates
- **Testing Framework**: Comprehensive test coverage using project standards
- **Quality Assurance**: Performance benchmarks and validation criteria

### 4. Quality Assurance
- **Pattern Verification**: Alignment with existing codebase conventions
- **Completeness Check**: All PRP components present and detailed
- **Technical Compliance**: Following project standards and best practices
- **Performance Targets**: Measurable success criteria and benchmarks

## Generated PRP Structure

The generated PRP follows the structured format defined in [`.claude/context/planning-methodology.md`](../../context/planning-methodology.md):

```markdown
# [Feature Name] - Product Requirements Prompt

## Overview
Brief description of what the plan accomplishes

**Created**: YYYY-MM-DD
**Status**: Draft
**Priority**: High/Medium/Low
**Estimated Duration**: X days/weeks
**Scope**: Specific boundaries

## Implementation Confidence Score: [8-10]/10
*Based on existing patterns and clear implementation path*

## Research Summary

### Existing EOL RAG Patterns Found
- **Package Dependencies**: [Relevant Python packages]
- **RAG Components**: [Document processors, indexers, searchers]
- **Redis Patterns**: [Vector operations, caching strategies]
- **Testing Patterns**: [Pytest fixtures, async test patterns]

### Key Dependencies Identified
- **Python Packages**: packages/eol-rag-context/
- **External Libraries**: redis[vector], fastmcp, typer
- **Redis Features**: Vector search, pipelines, TTL
- **LLM Providers**: Anthropic, OpenAI, local models

### Code Examples Found
```python
# From packages/eol-rag-context/src/document_processor.py
async def process_document(doc: Document) -> ProcessedDocument:
    """Actual pattern from codebase"""
    chunks = await self.chunk_content(doc.content)
    embeddings = await self.generate_embeddings(chunks)
    return ProcessedDocument(chunks, embeddings)
```

## Architecture Overview

### Python Component Structure
```
packages/eol-rag-context/
├── src/
│   ├── eol/
│   │   └── rag_context/
│   │       ├── document_processor.py
│   │       ├── embeddings.py
│   │       ├── indexer.py
│   │       └── redis_store.py
│   └── tests/
│       ├── test_document_processor.py
│       └── fixtures/
```

### Redis Infrastructure Design
- **Vector Index**: HNSW with cosine similarity
- **Caching Layer**: Semantic similarity cache
- **Connection Pool**: 50 max connections
- **Pipeline Operations**: Batch processing

## Implementation Tasks

### Phase 1: Python Foundation
- [ ] **Define data structures**
  - Create dataclasses with type hints
  - Define protocols for interfaces
  - Follow Python type annotation standards
  - Package: packages/eol-rag-context/

- [ ] **Set up package structure**
  - Configure pyproject.toml
  - Set up pytest configuration
  - Add type checking with mypy
  - Configure black and isort

### Phase 2: RAG Core Implementation
- [ ] **Document processing pipeline**
  ```python
  async def process_document(self, doc: Document) -> None:
      # Chunk content based on type
      chunks = await self.chunk_strategy.chunk(doc)
      
      # Generate embeddings
      embeddings = await self.embedding_provider.embed_batch(chunks)
      
      # Store in Redis
      await self.redis_store.index_chunks(chunks, embeddings)
  ```

- [ ] **Vector search implementation**
  ```python
  async def search_similar(self, query: str, k: int = 5) -> List[Document]:
      # Generate query embedding
      query_embedding = await self.embed(query)
      
      # Search Redis vector index
      results = await self.redis_store.vector_search(
          query_embedding, k=k
      )
      
      # Re-rank if needed
      return self.rerank_results(results)
  ```

### Phase 3: Caching Layer
- [ ] **Semantic cache implementation**
  - Similarity-based cache keys
  - TTL management (5min, 1hr, 24hr tiers)
  - Hit rate tracking (target >31%)
  - Cache warming strategies

### Phase 4: Testing & Validation
- [ ] **Unit tests with pytest**
  ```python
  @pytest.mark.asyncio
  async def test_document_processing(redis_store):
      processor = DocumentProcessor(redis_store)
      result = await processor.process(test_document)
      assert len(result.chunks) > 0
      assert result.embeddings is not None
  ```

- [ ] **Integration tests**
  - End-to-end RAG pipeline
  - Redis connection handling
  - Performance benchmarks

### Phase 5: Performance Optimization
- [ ] **Batch operations**
  - Pipeline Redis commands
  - Concurrent embedding generation
  - Async document processing

- [ ] **Performance targets**
  - Document indexing: >10 docs/sec
  - Vector search: <100ms latency
  - Cache hit rate: >31%

## Quality Gates

### Code Quality
```bash
# Format and lint
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/

# Test with coverage
pytest tests/ --cov=eol.rag_context --cov-fail-under=80
```

### Performance Validation
```python
# Benchmark tests
async def test_indexing_performance(benchmark):
    result = await benchmark(index_documents, test_docs)
    assert result.docs_per_sec > 10
```

## Success Metrics
- [ ] All tests passing with >80% coverage
- [ ] Performance targets met
- [ ] Type checking passes
- [ ] Documentation complete
- [ ] Integration tests successful

## Risk Mitigation
- **Performance Risk**: Start with small batches, optimize incrementally
- **Redis Memory**: Implement proper TTL and eviction policies
- **Embedding Costs**: Cache embeddings, batch operations
- **Type Safety**: Strict mypy configuration from start

## References
- `.claude/context/rag/` - RAG patterns and best practices
- `.claude/context/redis/` - Redis optimization strategies
- `.claude/context/python/` - Python async patterns
- `packages/eol-rag-context/` - Existing implementations
```

This PRP provides comprehensive context for implementing [feature] with confidence score [8-10]/10.