"""
Integration tests for all code examples in the tutorial.
Ensures all tutorial code is runnable and correct with real Redis v8.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
import json
import time
import numpy as np

from eol.rag_context import EOLRAGContextServer
from eol.rag_context.config import RAGConfig


@pytest.mark.integration
class TestTutorialExamples:
    """Test all code examples from TUTORIAL.md to ensure they work with real Redis."""
    
    @pytest.mark.asyncio
    async def test_basic_usage_starting_server(self, redis_store):
        """Test: Starting the Server (from Basic Usage section) with real Redis."""
        # With default configuration using real Redis
        server = EOLRAGContextServer()
        server.redis = redis_store  # Use real Redis from fixture
        server.indexer = None  # Will be set on first use
        server.cache = None
        server.graph = None
        server.watcher = None
        server._initialized = True
        
        assert server.redis is not None
        assert hasattr(server.redis, 'connect')
        
        # With custom configuration using real Redis
        config = RAGConfig()
        server2 = EOLRAGContextServer(config)
        server2.redis = redis_store  # Use real Redis from fixture
        server2._initialized = True
        assert server2.redis is not None
    
    @pytest.mark.asyncio
    async def test_indexing_single_file(self, redis_store, indexer_instance, temp_test_directory):
        """Test: Index a Single File (from Indexing Documents section) with real Redis."""
        server = EOLRAGContextServer()
        server.redis = redis_store  # Use real Redis
        server.indexer = indexer_instance  # Use real indexer with Redis
        server._initialized = True
        
        # Index a single file using real Redis
        test_file = temp_test_directory / "test.py"
        result = await server.index_directory(str(test_file))
        
        assert result['status'] == 'success'
        assert 'total_chunks' in result
        assert result['total_chunks'] > 0  # Should have indexed real content
        assert 'source_id' in result
        print(f"Indexed {result.get('total_chunks', 0)} chunks from {test_file}")
    
    @pytest.mark.asyncio
    async def test_indexing_directory(self, redis_store, indexer_instance, temp_test_directory):
        """Test: Index a Directory (from Indexing Documents section) with real Redis."""
        server = EOLRAGContextServer()
        server.redis = redis_store
        server.indexer = indexer_instance
        server._initialized = True
        
        # Index entire directory
        result = await server.index_directory(
            str(temp_test_directory),
            recursive=True,
            patterns=["*.py", "*.md", "*.yaml"],
            ignore=["__pycache__", ".git", "node_modules"]
        )
        
        assert result['status'] == 'success'
        assert result.get('indexed_files', 0) > 0
        assert result.get('total_chunks', 0) > 0
        assert 'source_id' in result
        
        print(f"Indexed {result['indexed_files']} files")
        print(f"Total chunks: {result['total_chunks']}")
        print(f"Source ID: {result['source_id']}")
    
    @pytest.mark.asyncio
    async def test_watch_for_changes(self, redis_store, file_watcher_instance, temp_test_directory):
        """Test: Watch for Changes (from Indexing Documents section) with real Redis."""
        server = EOLRAGContextServer()
        server.redis = redis_store
        server.watcher = file_watcher_instance
        server._initialized = True
        
        # Start watching a directory for changes
        watch_result = await server.watch_directory(
            str(temp_test_directory),
            patterns=["*.py", "*.md"],
            auto_index=True
        )
        
        assert watch_result['status'] == 'success'
        assert 'watch_id' in watch_result
        print(f"Watching with ID: {watch_result['watch_id']}")
        
        # Stop watching
        unwatch_result = await server.unwatch_directory(watch_result['watch_id'])
        assert unwatch_result['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_basic_search(self, redis_store, indexer_instance, temp_test_directory):
        """Test: Basic Search (from Searching & Retrieval section) with real Redis."""
        server = EOLRAGContextServer()
        server.redis = redis_store
        server.indexer = indexer_instance
        server._initialized = True
        
        # Index first
        await server.index_directory(str(temp_test_directory))
        
        # Search for relevant context
        results = await server.search_context(
            "How to implement authentication?",
            limit=5
        )
        
        assert isinstance(results, list)
        for result in results:
            assert 'score' in result
            assert 'content' in result
            assert 'metadata' in result
            print(f"Score: {result['score']:.2f}")
            print(f"Content: {result['content'][:200]}...")
            print(f"Source: {result['metadata'].get('source', 'unknown')}")
            print("---")
    
    @pytest.mark.asyncio
    async def test_hierarchical_search(self, redis_store, indexer_instance, temp_test_directory):
        """Test: Hierarchical Search (from Searching & Retrieval section) with real Redis."""
        server = EOLRAGContextServer()
        server.redis = redis_store
        server.indexer = indexer_instance
        server._initialized = True
        
        # Index first
        await server.index_directory(str(temp_test_directory))
        
        # Search at different hierarchy levels
        results = await server.search_context(
            "database connection",
            hierarchy_level=1,  # Concepts only
            limit=3
        )
        
        assert isinstance(results, list)
        
        # Get more detailed sections
        for concept in results[:1]:  # Just test with first concept
            sections = await server.search_context(
                "database connection",
                hierarchy_level=2,
                parent_id=concept.get('id'),
                limit=5
            )
            assert isinstance(sections, list)
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, redis_store, indexer_instance, temp_test_directory):
        """Test: Using Filters (from Searching & Retrieval section) with real Redis."""
        server = EOLRAGContextServer()
        server.redis = redis_store
        server.indexer = indexer_instance
        server._initialized = True
        
        # Index first
        await server.index_directory(str(temp_test_directory))
        
        # Search with metadata filters
        results = await server.search_context(
            "error handling",
            filters={
                "file_type": "python",
                "module": "auth",
                "last_modified": {"$gte": "2024-01-01"}
            },
            limit=10
        )
        
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_queries(self, redis_store, indexer_instance, knowledge_graph_instance, temp_test_directory):
        """Test: Knowledge Graph Queries (from Advanced Features section) with real Redis."""
        server = EOLRAGContextServer()
        server.redis = redis_store
        server.indexer = indexer_instance
        server.graph = knowledge_graph_instance
        server._initialized = True
        
        # Index first to build graph
        await server.index_directory(str(temp_test_directory))
        
        # Query the knowledge graph
        graph = await server.query_knowledge_graph(
            entity="TestClass",
            max_depth=2
        )
        
        assert 'entities' in graph
        assert 'relationships' in graph
        assert isinstance(graph['entities'], list)
        assert isinstance(graph['relationships'], list)
        
        print(f"Found {len(graph['entities'])} entities")
        print(f"Found {len(graph['relationships'])} relationships")
        
        # Visualize relationships
        for rel in graph['relationships']:
            print(f"{rel.get('source', 'unknown')} --{rel.get('type', 'unknown')}--> {rel.get('target', 'unknown')}")
    
    @pytest.mark.asyncio
    async def test_semantic_caching(self, redis_store, indexer_instance, semantic_cache_instance, temp_test_directory):
        """Test: Semantic Caching (from Advanced Features section) with real Redis."""
        server = EOLRAGContextServer()
        server.redis = redis_store
        server.indexer = indexer_instance
        server.cache = semantic_cache_instance
        server._initialized = True
        
        # Index first
        await server.index_directory(str(temp_test_directory))
        
        # Enable semantic caching for repeated queries
        await server.optimize_context(target_hit_rate=0.31)
        
        # First query (cache miss)
        start = time.time()
        results1 = await server.search_context("user authentication flow")
        time1 = time.time() - start
        print(f"First query: {time1:.2f}s")
        
        # Similar query (potential cache hit)
        start = time.time()
        results2 = await server.search_context("authentication process for users")
        time2 = time.time() - start
        print(f"Cached query: {time2:.2f}s")
        
        # Get cache statistics
        stats = await server.get_stats()
        assert 'cache' in stats
        print(f"Cache hit rate: {stats['cache'].get('hit_rate', 0):.2%}")
    
    @pytest.mark.asyncio
    async def test_context_windows_management(self, redis_store, indexer_instance, temp_test_directory):
        """Test: Context Windows Management (from Advanced Features section) with real Redis."""
        server = EOLRAGContextServer()
        server.redis = redis_store
        server.indexer = indexer_instance
        server._initialized = True
        
        # Index first
        await server.index_directory(str(temp_test_directory))
        
        # Get optimized context for LLM
        context = await server.get_context(
            "context://implement payment processing",
            max_tokens=4000,
            include_hierarchy=True
        )
        
        assert isinstance(context, list)
        
        # Format for LLM consumption
        formatted = "\n\n".join([
            f"## {doc.get('metadata', {}).get('header', 'Context')}\n{doc.get('content', '')}"
            for doc in context
        ])
        
        assert isinstance(formatted, str)
    
    @pytest.mark.asyncio
    async def test_code_assistant_example(self, redis_store, indexer_instance, temp_test_directory):
        """Test: Code Assistant Example (from Integration Examples section) with real Redis."""
        async def code_assistant():
            # Initialize RAG server with real Redis
            server = EOLRAGContextServer()
            server.redis = redis_store
            server.indexer = indexer_instance
            server._initialized = True
            
            # Index your codebase
            await server.index_directory(
                str(temp_test_directory),
                patterns=["*.py", "*.js", "*.md"]
            )
            
            # User query
            query = "How do I add a new API endpoint?"
            
            # Get relevant context
            context = await server.search_context(query, limit=5)
            
            # Build prompt for LLM
            prompt = "Based on the following context, answer the question.\n\n"
            prompt += "Context:\n"
            for ctx in context:
                prompt += f"- {ctx['content'][:500]}...\n"
            prompt += f"\nQuestion: {query}\nAnswer:"
            
            # Verify prompt was built
            assert isinstance(prompt, str)
            assert query in prompt
            
            return context
        
        # Run the assistant
        context = await code_assistant()
        assert isinstance(context, list)
    
    @pytest.mark.asyncio
    async def test_documentation_search_example(self, redis_store, indexer_instance, temp_test_directory):
        """Test: Documentation Search Example (from Integration Examples section) with real Redis."""
        async def search_docs(query: str):
            server = EOLRAGContextServer()
            server.redis = redis_store
            server.indexer = indexer_instance
            server._initialized = True
            
            # Index first
            await server.index_directory(str(temp_test_directory))
            
            # Search only markdown files
            results = await server.search_context(
                query,
                filters={"file_type": "markdown"},
                hierarchy_level=2,  # Section level
                limit=10
            )
            
            # Group by document
            docs = {}
            for result in results:
                source = result.get('metadata', {}).get('source', 'unknown')
                if source not in docs:
                    docs[source] = []
                docs[source].append(result)
            
            # Display results
            for doc, sections in docs.items():
                print(f"\n📄 {doc}")
                for section in sections:
                    print(f"  - {section.get('metadata', {}).get('header', 'Section')}")
                    print(f"    {section.get('content', '')[:100]}...")
            
            return docs
        
        # Test the search
        docs = await search_docs("test")
        assert isinstance(docs, dict)
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, redis_store, indexer_instance, temp_test_directory):
        """Test: Batch Operations for Performance (from Best Practices section) with real Redis."""
        server = EOLRAGContextServer()
        server.redis = redis_store
        server.indexer = indexer_instance
        server._initialized = True
        
        # Batch operations for better performance
        async def batch_index(server, files):
            batch_size = 10
            
            for i in range(0, len(files), batch_size):
                batch = files[i:i + batch_size]
                tasks = [
                    server.index_directory(str(f))
                    for f in batch
                ]
                results = await asyncio.gather(*tasks)
                print(f"Indexed batch {i//batch_size + 1}")
                assert all(r['status'] == 'success' for r in results)
            
            return results
        
        # Get files to index
        files = list(temp_test_directory.glob("*"))[:3]  # Limit for testing
        if files:
            results = await batch_index(server, files)
            assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_context_optimization_for_llm(self):
        """Test: Context Window Management (from Best Practices section)."""
        def optimize_context_for_llm(contexts, max_tokens=4000):
            """Optimize context to fit in LLM window."""
            
            # Sort by relevance
            contexts.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Track token count (rough estimate)
            result = []
            token_count = 0
            
            for ctx in contexts:
                # Estimate tokens (roughly 4 chars = 1 token)
                ctx_tokens = len(ctx.get('content', '')) // 4
                
                if token_count + ctx_tokens > max_tokens:
                    # Truncate if needed
                    remaining = max_tokens - token_count
                    if remaining > 100:  # Worth including partial
                        truncated = ctx['content'][:remaining * 4]
                        ctx['content'] = truncated
                        result.append(ctx)
                    break
                
                result.append(ctx)
                token_count += ctx_tokens
            
            return result
        
        # Test with sample contexts
        contexts = [
            {'content': 'x' * 1000, 'score': 0.9},
            {'content': 'y' * 2000, 'score': 0.8},
            {'content': 'z' * 3000, 'score': 0.7},
            {'content': 'a' * 4000, 'score': 0.6},
        ]
        
        optimized = optimize_context_for_llm(contexts, max_tokens=1000)
        
        # Verify optimization
        total_chars = sum(len(ctx['content']) for ctx in optimized)
        assert total_chars <= 4000  # 1000 tokens * 4 chars
        assert len(optimized) > 0
        assert optimized[0]['score'] >= optimized[-1]['score']  # Sorted by score
    
    @pytest.mark.asyncio
    async def test_health_check_example(self, redis_store, indexer_instance, semantic_cache_instance, knowledge_graph_instance):
        """Test: Health Check Example (from Troubleshooting section) with real Redis."""
        async def health_check():
            server = EOLRAGContextServer()
            server.redis = redis_store
            server.indexer = indexer_instance
            server.cache = semantic_cache_instance
            server.graph = knowledge_graph_instance
            server._initialized = True
            
            try:
                stats = await server.get_stats()
                
                print("✅ Server Status: Healthy")
                print(f"📊 Documents: {stats['indexer'].get('total_documents', 0)}")
                print(f"📦 Chunks: {stats['indexer'].get('total_chunks', 0)}")
                print(f"💾 Cache Hit Rate: {stats['cache'].get('hit_rate', 0):.1%}")
                print(f"🔗 Graph Nodes: {stats['graph'].get('nodes', 0)}")
                
                return True
                
            except Exception as e:
                print(f"❌ Health Check Failed: {e}")
                return False
        
        # Run health check
        healthy = await health_check()
        assert healthy is True
    
    @pytest.mark.asyncio
    async def test_smart_indexing_strategy(self, redis_store, indexer_instance, file_watcher_instance, temp_test_directory):
        """Test: Smart Indexing Strategy (from Best Practices section) with real Redis."""
        server = EOLRAGContextServer()
        server.redis = redis_store
        server.indexer = indexer_instance
        server.watcher = file_watcher_instance
        server._initialized = True
        
        # Index in priority order
        async def smart_indexing(server, project_path):
            results = []
            
            # Check if docs directory exists
            docs_path = project_path / "docs"
            if not docs_path.exists():
                docs_path.mkdir()
                (docs_path / "README.md").write_text("# Documentation\nTest docs")
            
            # 1. Index critical documentation first
            result = await server.index_directory(
                str(docs_path),
                priority=1
            )
            results.append(result)
            
            # 2. Index main source code
            src_path = project_path / "src"
            if not src_path.exists():
                src_path.mkdir()
                (src_path / "main.py").write_text("def main(): pass")
            
            result = await server.index_directory(
                str(src_path),
                priority=2,
                patterns=["*.py", "*.js"]
            )
            results.append(result)
            
            # 3. Index tests and examples (if they exist)
            tests_path = project_path / "tests"
            if tests_path.exists():
                result = await server.index_directory(
                    str(tests_path),
                    priority=3
                )
                results.append(result)
            
            # 4. Watch for changes
            watch_result = await server.watch_directory(
                str(project_path),
                auto_index=True
            )
            
            # Clean up watch
            await server.unwatch_directory(watch_result['watch_id'])
            
            return results
        
        results = await smart_indexing(server, temp_test_directory)
        assert len(results) >= 2  # At least docs and src
        assert all(r['status'] == 'success' for r in results)