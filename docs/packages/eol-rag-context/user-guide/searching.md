# Search & Retrieval

Master the art of semantic search and intelligent context retrieval with EOL RAG Context. This guide covers everything from basic queries to advanced search strategies that leverage the full power of vector similarity and hierarchical organization.

## Overview

EOL RAG Context transforms traditional keyword search into intelligent semantic retrieval. Instead of matching exact words, it understands meaning, context, and relationships to find the most relevant information.

### Key Features

- **Semantic Understanding**: Find content by meaning, not just keywords
- **Hierarchical Search**: Query at concept, section, or chunk level
- **Vector Similarity**: Mathematical relevance scoring
- **Context Assembly**: Intelligent result organization
- **Real-time Performance**: Sub-100ms search with caching

## Basic Search

### Simple Queries

Start with straightforward search requests:

```python
import asyncio
from eol.rag_context import EOLRAGContextServer
from eol.rag_context.server import SearchContextRequest

async def basic_search():
    server = EOLRAGContextServer()
    await server.initialize()

    # Simple search query
    request = SearchContextRequest(
        query="How to initialize a database connection?",
        max_results=5,
        similarity_threshold=0.7
    )

    results = await server.search_context(request, None)

    # Display results
    for i, result in enumerate(results['results'], 1):
        print(f"\n🔍 Result {i}:")
        print(f"   File: {result['source_path']}")
        print(f"   Relevance: {result['similarity']:.3f}")
        print(f"   Content: {result['content'][:200]}...")

asyncio.run(basic_search())
```

### Understanding Results

Each search result contains rich metadata:

```python
{
    'content': str,           # The matched content
    'source_path': str,       # Original file path
    'chunk_type': str,        # Type: function, class, paragraph, etc.
    'similarity': float,      # Relevance score (0.0-1.0)
    'metadata': {
        'lines': (int, int),  # Line numbers in source
        'language': str,      # Programming language
        'section': str,       # Document section/header
        'chunk_index': int,   # Position in document
        'file_size': int,     # Original file size
        'modified': float,    # Last modification time
    }
}
```

### Similarity Scores

Understanding relevance indicators:

| Score Range | Interpretation | Use Case |
|------------|---------------|----------|
| **0.95-1.0** | Exact/near-exact match | Perfect matches, duplicates |
| **0.85-0.95** | Highly relevant | Direct answers, implementations |
| **0.75-0.85** | Very relevant | Related concepts, examples |
| **0.65-0.75** | Moderately relevant | Background info, context |
| **Below 0.65** | Low relevance | May not be useful |

## Advanced Queries

### Semantic vs Keyword Search

Compare traditional keyword matching with semantic understanding:

```python
async def compare_search_types():
    server = EOLRAGContextServer()
    await server.initialize()

    # Traditional keyword-style query
    keyword_query = "database connection pool settings"

    # Semantic query expressing intent
    semantic_query = "How do I configure connection pooling for better database performance?"

    # Natural language query
    natural_query = "I want to optimize database connections in my application"

    queries = [
        ("Keyword", keyword_query),
        ("Semantic", semantic_query),
        ("Natural", natural_query)
    ]

    for query_type, query in queries:
        print(f"\n{query_type} Query: {query}")
        print("=" * 60)

        request = SearchContextRequest(
            query=query,
            max_results=3,
            similarity_threshold=0.7
        )

        results = await server.search_context(request, None)

        for result in results['results']:
            print(f"  📄 {result['source_path']} (score: {result['similarity']:.3f})")
```

### Multi-Concept Queries

Search for multiple related concepts:

```python
async def multi_concept_search():
    # Query combining multiple concepts
    complex_queries = [
        "error handling AND database transactions",
        "authentication OR authorization in web APIs",
        "performance optimization for large datasets",
        "testing strategies for microservices architecture"
    ]

    for query in complex_queries:
        print(f"\n🔍 Complex Query: {query}")

        request = SearchContextRequest(
            query=query,
            max_results=5,
            similarity_threshold=0.75,
            include_metadata=True
        )

        results = await server.search_context(request, None)

        # Group results by concept
        concepts = {}
        for result in results['results']:
            chunk_type = result['chunk_type']
            if chunk_type not in concepts:
                concepts[chunk_type] = []
            concepts[chunk_type].append(result)

        for concept, items in concepts.items():
            print(f"  📂 {concept.title()}: {len(items)} results")
```

## Hierarchical Search

### Search Levels

Target different granularities of information:

```python
async def hierarchical_search():
    server = EOLRAGContextServer()
    await server.initialize()

    query = "user authentication implementation"

    # Search at different hierarchy levels
    levels = ['concept', 'section', 'chunk']

    for level in levels:
        print(f"\n🎯 {level.title()} Level Search:")
        print("-" * 40)

        request = SearchContextRequest(
            query=query,
            max_results=3,
            search_level=level  # concept/section/chunk
        )

        results = await server.search_context(request, None)

        for result in results['results']:
            print(f"  📍 {result['metadata'].get('section', 'N/A')}")
            print(f"     Score: {result['similarity']:.3f}")
            print(f"     Type: {result['chunk_type']}")
            print(f"     Preview: {result['content'][:100]}...")
```

**When to Use Each Level:**

- **Concept**: High-level overview, architectural decisions
- **Section**: Specific implementations, detailed explanations
- **Chunk**: Exact code snippets, precise details

### Cross-Level Synthesis

Combine results from multiple hierarchy levels:

```python
async def cross_level_search():
    query = "How does caching work in the application?"

    # Gather context from all levels
    all_results = {}

    for level in ['concept', 'section', 'chunk']:
        request = SearchContextRequest(
            query=query,
            max_results=2,
            search_level=level
        )

        results = await server.search_context(request, None)
        all_results[level] = results['results']

    # Synthesize comprehensive answer
    print("🎯 Comprehensive Context:")
    print("\n📋 Conceptual Overview:")
    for result in all_results['concept']:
        print(f"   • {result['content'][:150]}...")

    print("\n🔧 Implementation Details:")
    for result in all_results['section']:
        print(f"   • {result['content'][:150]}...")

    print("\n💻 Code Examples:")
    for result in all_results['chunk']:
        print(f"   • {result['content'][:150]}...")
```

## Filtering and Ranking

### Advanced Filtering

Filter results by various criteria:

```python
async def filtered_search():
    server = EOLRAGContextServer()
    await server.initialize()

    request = SearchContextRequest(
        query="database optimization techniques",
        max_results=10,

        # Content filters
        filters={
            'file_types': ['.py', '.md'],           # Only Python and Markdown
            'languages': ['python'],                # Only Python code
            'date_range': {                         # Only recent files
                'after': '2024-01-01',
                'before': '2024-12-31'
            },
            'file_size': {                          # File size constraints
                'min_kb': 1,
                'max_kb': 1000
            },
            'chunk_types': [                        # Specific content types
                'function', 'class', 'documentation'
            ]
        },

        # Ranking preferences
        ranking={
            'semantic_weight': 0.6,     # Semantic similarity importance
            'recency_weight': 0.2,      # How much to favor recent files
            'structure_weight': 0.2,    # Document structure importance
        }
    )

    results = await server.search_context(request, None)

    for result in results['results']:
        metadata = result['metadata']
        print(f"📄 {result['source_path']}")
        print(f"   Language: {metadata.get('language', 'N/A')}")
        print(f"   Modified: {metadata.get('modified', 'N/A')}")
        print(f"   Type: {result['chunk_type']}")
        print(f"   Score: {result['similarity']:.3f}")
```

### Custom Ranking Strategies

Implement domain-specific ranking:

```python
async def custom_ranking():
    # Define ranking strategy for code search
    code_ranking = {
        'semantic_weight': 0.5,      # Base semantic relevance
        'code_quality_weight': 0.2,  # Favor well-documented code
        'usage_weight': 0.15,        # Favor frequently referenced code
        'recency_weight': 0.15,      # Slight preference for recent code
    }

    # Define ranking strategy for documentation search
    docs_ranking = {
        'semantic_weight': 0.7,      # Higher semantic weight
        'completeness_weight': 0.2,  # Favor complete explanations
        'recency_weight': 0.1,       # Less emphasis on recency
    }

    query = "how to implement user authentication"

    # Search with code-focused ranking
    code_request = SearchContextRequest(
        query=query,
        max_results=5,
        filters={'file_types': ['.py', '.js', '.go']},
        ranking=code_ranking
    )

    # Search with documentation-focused ranking
    docs_request = SearchContextRequest(
        query=query,
        max_results=5,
        filters={'file_types': ['.md', '.rst', '.txt']},
        ranking=docs_ranking
    )

    code_results = await server.search_context(code_request, None)
    docs_results = await server.search_context(docs_request, None)

    print("🖥️  Code-focused results:")
    for result in code_results['results']:
        print(f"   {result['source_path']} ({result['similarity']:.3f})")

    print("\n📚 Documentation-focused results:")
    for result in docs_results['results']:
        print(f"   {result['source_path']} ({result['similarity']:.3f})")
```

## Context Assembly

### Intelligent Context Building

Assemble coherent context from multiple search results:

```python
async def assemble_context():
    server = EOLRAGContextServer()
    await server.initialize()

    query = "complete user registration flow"

    # Search for comprehensive context
    request = SearchContextRequest(
        query=query,
        max_results=10,
        assemble_context=True,        # Enable context assembly
        context_strategy='hierarchical',  # Assembly strategy
        max_context_size=4000,       # Target context size
        include_surrounding=True,     # Include surrounding context
    )

    results = await server.search_context(request, None)

    # Results include assembled context
    assembled_context = results.get('assembled_context', '')
    context_sources = results.get('context_sources', [])

    print("🎯 Assembled Context:")
    print("=" * 50)
    print(assembled_context)

    print(f"\n📚 Sources ({len(context_sources)}):")
    for source in context_sources:
        print(f"   • {source['file']} (lines {source['lines']})")
```

### Context Optimization

Optimize context for specific use cases:

```python
async def optimize_context():
    # Context for code explanation
    code_context_config = {
        'strategy': 'code_focused',
        'include_imports': True,       # Include relevant imports
        'include_docstrings': True,    # Include function documentation
        'include_examples': True,      # Include usage examples
        'max_context_size': 3000,
    }

    # Context for troubleshooting
    debug_context_config = {
        'strategy': 'problem_solving',
        'include_error_handling': True,  # Include error handling code
        'include_logging': True,         # Include logging statements
        'include_tests': True,           # Include relevant tests
        'max_context_size': 5000,
    }

    query = "database connection timeout errors"

    # Get debugging-optimized context
    debug_request = SearchContextRequest(
        query=query,
        context_config=debug_context_config
    )

    results = await server.search_context(debug_request, None)

    print("🐛 Debug Context:")
    print(results['assembled_context'])
```

## Query Optimization

### Query Enhancement

Improve query effectiveness:

```python
async def enhance_queries():
    server = EOLRAGContextServer()
    await server.initialize()

    # Original vague query
    vague_query = "database stuff"

    # Enhanced specific queries
    enhanced_queries = [
        "database connection configuration and pooling",
        "SQL query optimization and performance tuning",
        "database migration and schema management",
        "database transaction handling and error recovery"
    ]

    print(f"Original query: '{vague_query}'")
    basic_request = SearchContextRequest(query=vague_query, max_results=3)
    basic_results = await server.search_context(basic_request, None)
    print(f"Results: {len(basic_results['results'])} (avg score: {sum(r['similarity'] for r in basic_results['results'])/len(basic_results['results']):.3f})")

    print("\nEnhanced queries:")
    for enhanced_query in enhanced_queries:
        request = SearchContextRequest(query=enhanced_query, max_results=3)
        results = await server.search_context(request, None)
        avg_score = sum(r['similarity'] for r in results['results'])/len(results['results']) if results['results'] else 0
        print(f"  '{enhanced_query}' → {len(results['results'])} results (avg: {avg_score:.3f})")
```

### Query Expansion

Automatically expand queries with related terms:

```python
async def query_expansion():
    server = EOLRAGContextServer()
    await server.initialize()

    base_query = "user authentication"

    # Enable query expansion
    request = SearchContextRequest(
        query=base_query,
        max_results=10,
        expand_query=True,           # Enable expansion
        expansion_terms=5,           # Number of related terms to add
        expansion_strategy='semantic', # How to find related terms
    )

    results = await server.search_context(request, None)

    expanded_query = results.get('expanded_query', base_query)
    expansion_terms = results.get('expansion_terms', [])

    print(f"Original: {base_query}")
    print(f"Expanded: {expanded_query}")
    print(f"Added terms: {', '.join(expansion_terms)}")

    print(f"\nResults with expansion: {len(results['results'])}")
    for result in results['results'][:3]:
        print(f"  📄 {result['source_path']} ({result['similarity']:.3f})")
```

## Performance Optimization

### Search Performance

Optimize search speed and accuracy:

```python
async def optimize_search_performance():
    server = EOLRAGContextServer()
    await server.initialize()

    # Performance-optimized search configuration
    fast_config = {
        'max_results': 5,              # Limit result set
        'similarity_threshold': 0.8,   # Higher threshold = fewer results
        'use_cache': True,             # Enable semantic caching
        'cache_ttl': 3600,            # Cache for 1 hour
        'parallel_search': True,       # Search multiple indexes parallel
        'early_termination': True,     # Stop when enough results found
    }

    import time

    query = "error handling in microservices"

    # Benchmark search performance
    start_time = time.time()

    request = SearchContextRequest(
        query=query,
        **fast_config
    )

    results = await server.search_context(request, None)

    search_time = (time.time() - start_time) * 1000  # Convert to ms

    print(f"⚡ Search Performance:")
    print(f"   Query: {query}")
    print(f"   Results: {len(results['results'])}")
    print(f"   Search time: {search_time:.1f}ms")
    print(f"   Cache hit: {results.get('cache_hit', False)}")

    # Display results with timing info
    for i, result in enumerate(results['results'], 1):
        print(f"   {i}. {result['source_path']} ({result['similarity']:.3f})")
```

### Caching Strategies

Leverage semantic caching for better performance:

```python
async def caching_strategies():
    server = EOLRAGContextServer()
    await server.initialize()

    # Configure semantic caching
    cache_config = {
        'enabled': True,
        'similarity_threshold': 0.95,  # Cache very similar queries
        'ttl_seconds': 1800,          # 30 minute TTL
        'max_cache_size': 1000,       # Maximum cached queries
        'adaptive_threshold': True,    # Auto-adjust threshold
    }

    await server.configure_cache(**cache_config)

    # Test queries that should benefit from caching
    similar_queries = [
        "how to implement user authentication",
        "user authentication implementation guide",
        "implementing authentication for users",
        "user auth implementation tutorial"
    ]

    print("🚀 Testing semantic cache:")

    for i, query in enumerate(similar_queries):
        start_time = time.time()

        request = SearchContextRequest(query=query, max_results=3)
        results = await server.search_context(request, None)

        search_time = (time.time() - start_time) * 1000
        cache_hit = results.get('cache_hit', False)

        print(f"  Query {i+1}: {search_time:.1f}ms {'(cached)' if cache_hit else '(new)'}")

    # Check cache statistics
    cache_stats = await server.get_cache_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"   Hit rate: {cache_stats['hit_rate']:.1%}")
    print(f"   Total queries: {cache_stats['total_queries']}")
    print(f"   Cache size: {cache_stats['cache_size']}")
```

## Search Analytics

### Query Analytics

Track and analyze search patterns:

```python
async def search_analytics():
    server = EOLRAGContextServer()
    await server.initialize()

    # Enable search analytics
    await server.enable_search_analytics(
        track_queries=True,
        track_results=True,
        track_performance=True,
        export_metrics=True
    )

    # Simulate various search patterns
    search_patterns = [
        ("Quick lookups", ["config", "setup", "install"]),
        ("How-to queries", ["how to deploy", "how to configure", "how to test"]),
        ("Troubleshooting", ["error", "fix", "debug", "problem"]),
        ("Code search", ["function", "class", "implementation", "algorithm"])
    ]

    for pattern_name, queries in search_patterns:
        print(f"\n🔍 Simulating: {pattern_name}")

        for query in queries:
            request = SearchContextRequest(
                query=query,
                max_results=3,
                track_analytics=True,  # Enable tracking for this query
                pattern_label=pattern_name  # Label for analytics
            )

            results = await server.search_context(request, None)
            print(f"   '{query}' → {len(results['results'])} results")

    # Get analytics report
    analytics = await server.get_search_analytics()

    print(f"\n📊 Search Analytics Summary:")
    print(f"   Total queries: {analytics['total_queries']}")
    print(f"   Average results per query: {analytics['avg_results_per_query']:.1f}")
    print(f"   Average search time: {analytics['avg_search_time_ms']:.1f}ms")
    print(f"   Most common query types:")

    for query_type, count in analytics['query_types'].items():
        print(f"      {query_type}: {count} queries")
```

### Result Quality Metrics

Monitor and improve search result quality:

```python
async def quality_metrics():
    server = EOLRAGContextServer()
    await server.initialize()

    # Enable quality tracking
    await server.enable_quality_tracking(
        track_relevance=True,
        track_user_feedback=True,
        track_result_usage=True
    )

    # Search with quality tracking
    request = SearchContextRequest(
        query="database connection pooling best practices",
        max_results=5,
        track_quality=True
    )

    results = await server.search_context(request, None)

    # Analyze result quality
    quality_metrics = results.get('quality_metrics', {})

    print("📊 Result Quality Metrics:")
    print(f"   Relevance distribution:")
    for score_range, count in quality_metrics.get('relevance_distribution', {}).items():
        print(f"      {score_range}: {count} results")

    print(f"   Diversity score: {quality_metrics.get('diversity_score', 0):.3f}")
    print(f"   Coverage score: {quality_metrics.get('coverage_score', 0):.3f}")
    print(f"   Freshness score: {quality_metrics.get('freshness_score', 0):.3f}")

    # Simulate user feedback
    for i, result in enumerate(results['results'][:3]):
        # Simulate relevance feedback (in real use, this comes from users)
        relevance_score = 0.9 if result['similarity'] > 0.8 else 0.6

        await server.record_result_feedback(
            query_id=results['query_id'],
            result_id=result['id'],
            relevance_score=relevance_score,
            user_clicked=True,
            user_helpful=relevance_score > 0.7
        )
```

## Troubleshooting

### Common Search Issues

**No Results Found:**

```python
async def diagnose_no_results():
    query = "machine learning algorithms"

    # Diagnostic steps
    print(f"🔍 Diagnosing: '{query}'")

    # Step 1: Check if content exists
    all_content = await server.get_indexed_content_summary()
    print(f"   Indexed documents: {all_content['document_count']}")
    print(f"   Total chunks: {all_content['chunk_count']}")

    # Step 2: Try broader query
    broad_request = SearchContextRequest(
        query="algorithm",  # Broader term
        similarity_threshold=0.6,  # Lower threshold
        max_results=10
    )
    broad_results = await server.search_context(broad_request, None)
    print(f"   Broader search results: {len(broad_results['results'])}")

    # Step 3: Check embedding quality
    embedding_info = await server.get_embedding_info(query)
    print(f"   Query embedding dimension: {len(embedding_info['embedding'])}")
    print(f"   Similar indexed terms: {embedding_info['similar_terms'][:5]}")

    # Step 4: Suggest query improvements
    suggestions = await server.suggest_query_improvements(query)
    print(f"   Suggested queries: {suggestions}")
```

**Low Quality Results:**

```python
async def improve_result_quality():
    query = "database optimization"

    # Get initial results
    basic_request = SearchContextRequest(query=query, max_results=5)
    basic_results = await server.search_context(basic_request, None)

    avg_score = sum(r['similarity'] for r in basic_results['results']) / len(basic_results['results'])
    print(f"Basic search average score: {avg_score:.3f}")

    if avg_score < 0.75:  # Low quality threshold
        print("🔧 Improving result quality:")

        # Try query expansion
        expanded_request = SearchContextRequest(
            query=query,
            expand_query=True,
            similarity_threshold=0.8,  # Higher threshold
            max_results=5
        )
        expanded_results = await server.search_context(expanded_request, None)

        # Try hierarchical search
        concept_request = SearchContextRequest(
            query=query,
            search_level='concept',
            max_results=5
        )
        concept_results = await server.search_context(concept_request, None)

        # Compare improvements
        expanded_avg = sum(r['similarity'] for r in expanded_results['results']) / len(expanded_results['results'])
        concept_avg = sum(r['similarity'] for r in concept_results['results']) / len(concept_results['results'])

        print(f"   Expanded query: {expanded_avg:.3f} average score")
        print(f"   Concept search: {concept_avg:.3f} average score")

        # Recommend best approach
        if expanded_avg > concept_avg and expanded_avg > avg_score:
            print("   ✅ Recommendation: Use query expansion")
        elif concept_avg > avg_score:
            print("   ✅ Recommendation: Use concept-level search")
        else:
            print("   ⚠️ Consider reindexing with better chunking strategy")
```

## Best Practices

### Query Design

**Effective Query Patterns:**

- ✅ "How to implement OAuth authentication in Python"
- ✅ "Database connection pooling configuration examples"
- ✅ "Error handling patterns for microservices"
- ❌ "auth stuff"
- ❌ "database"
- ❌ "code"

**Query Optimization Tips:**

1. **Be Specific**: Include context and intent
2. **Use Natural Language**: Write how you'd ask a colleague
3. **Include Domain Terms**: Use relevant technical vocabulary
4. **Specify Format**: "examples", "tutorial", "configuration"

### Search Strategy

**Progressive Search Refinement:**

1. Start with broad concept-level search
2. Narrow down to section-level for details
3. Use chunk-level for exact implementations
4. Combine results for comprehensive understanding

**Result Validation:**

1. Check similarity scores (aim for >0.75)
2. Verify result diversity (avoid duplicates)
3. Confirm temporal relevance (recent vs historical)
4. Validate source quality (well-documented vs sketchy)

## Next Steps

Now that you've mastered search and retrieval:

1. **[Advanced Features](advanced-features.md)** - Explore knowledge graphs and performance optimization
2. **[MCP Integration](integrations.md)** - Connect with Claude Desktop for seamless experience
3. **[Examples](../examples/)** - See real-world search patterns and use cases
4. **[API Reference](../api-reference/)** - Deep dive into search API capabilities

Ready to explore the most powerful features? Continue with **[Advanced Features](advanced-features.md)**.
