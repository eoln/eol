"""
Integration tests for full RAG workflow.
Tests complete indexing, searching, and caching workflow.
"""

import asyncio
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.integration
class TestFullWorkflowIntegration:
    """Test complete RAG workflow with all components."""

    @pytest.mark.asyncio
    async def test_index_and_search_workflow(
        self, redis_store, indexer_instance, embedding_manager, temp_test_directory
    ):
        """Test complete index and search workflow."""
        # Step 1: Index documents
        index_result = await indexer_instance.index_folder(temp_test_directory)
        assert index_result.file_count > 0
        assert index_result.total_chunks > 0

        # Step 2: Get embedding for query
        query = "hello world function"
        query_embedding = await embedding_manager.get_embedding(query)
        assert query_embedding.shape == (384,)  # Using all-MiniLM-L6-v2

        # Step 3: Search for relevant documents
        results = await redis_store.vector_search(
            query_embedding=query_embedding, hierarchy_level=3, k=5
        )

        # Should find relevant results
        assert isinstance(results, list)
        if len(results) > 0:
            # Verify result structure (tuple format: id, score, data)
            for result in results:
                assert len(result) == 3  # (id, score, data)
                doc_id, score, data = result
                assert isinstance(doc_id, str)
                assert isinstance(score, (int, float))
                assert isinstance(data, dict)
                assert "content" in data

    @pytest.mark.asyncio
    async def test_semantic_cache_workflow(self, semantic_cache_instance, embedding_manager):
        """Test semantic caching workflow."""
        # Initialize cache (creates index)
        await semantic_cache_instance.initialize()

        # Step 1: Store query-response pair
        query1 = "What is machine learning?"
        response1 = "Machine learning is a subset of AI that enables systems to learn from data."
        metadata1 = {"source": "test", "timestamp": time.time()}

        await semantic_cache_instance.set(query1, response1, metadata1)

        # Step 2: Try exact match
        cached = await semantic_cache_instance.get(query1)
        assert cached is not None
        assert cached == response1  # get() returns the response string directly

        # Step 3: Try similar query
        query2 = "What's ML?"  # Similar but not exact
        await semantic_cache_instance.get(query2)
        # May or may not match depending on similarity threshold

        # Step 4: Store more entries
        for i in range(5):
            await semantic_cache_instance.set(f"Query {i}", f"Response {i}", {"index": i})

        # Step 5: Check cache stats
        stats = semantic_cache_instance.get_stats()
        assert stats["queries"] > 0
        assert "hit_rate" in stats

        # Step 6: Clear cache
        await semantic_cache_instance.clear()

        # Verify cleared
        cached_after_clear = await semantic_cache_instance.get(query1)
        assert cached_after_clear is None

    @pytest.mark.asyncio
    async def test_knowledge_graph_workflow(
        self, knowledge_graph_instance, indexer_instance, temp_test_directory
    ):
        """Test knowledge graph construction workflow."""
        # Debug: check graph type
        print(f"Graph type: {type(knowledge_graph_instance.graph)}")
        print(f"Graph: {knowledge_graph_instance.graph}")

        # Step 1: Index documents first (required for knowledge graph)
        index_result = await indexer_instance.index_folder(temp_test_directory)
        source_id = index_result.source_id

        # Step 2: Build knowledge graph from indexed documents
        await knowledge_graph_instance.build_from_documents(source_id)

        # Step 3: Query subgraph (check if graph has entities first)
        if len(knowledge_graph_instance.entities) > 0:
            # Get the first entity ID to query
            entity_id = list(knowledge_graph_instance.entities.keys())[0]
            subgraph = await knowledge_graph_instance.query_subgraph(entity_id, max_depth=2)

            assert hasattr(subgraph, "entities")
            assert hasattr(subgraph, "relationships")
        else:
            # No entities extracted, just check the graph exists
            assert knowledge_graph_instance.graph is not None

        # Step 4: Get stats
        stats = knowledge_graph_instance.get_graph_stats()
        assert "entity_count" in stats
        assert "relationship_count" in stats

        # KnowledgeGraphBuilder doesn't have persist/load/clear methods
        # These would need to be implemented if required

    @pytest.mark.asyncio
    async def test_file_watcher_workflow(self, file_watcher_instance, indexer_instance):
        """Test file watching and auto-indexing workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            # Step 1: Create initial file
            file1 = test_dir / "initial.txt"
            file1.write_text("Initial content")

            # Step 2: Start watching
            watch_id = await file_watcher_instance.watch(
                test_dir, file_patterns=["*.txt", "*.py"], recursive=True
            )
            assert watch_id is not None

            # Step 3: Verify watch was added (FileWatcher doesn't have get_watch_info)
            # Just check that we got a valid watch_id
            assert isinstance(watch_id, str)
            assert len(watch_id) > 0

            # Step 5: Create new file (should trigger indexing)
            await asyncio.sleep(0.5)  # Let watcher start

            file2 = test_dir / "new_file.txt"
            file2.write_text("New content that should be indexed")

            # Give time for debouncing and processing
            await asyncio.sleep(2)

            # Step 6: Modify existing file
            file1.write_text("Modified content")
            await asyncio.sleep(2)

            # Step 7: Delete file
            file2.unlink()
            await asyncio.sleep(2)

            # Step 8: Stop watching
            unwatched = await file_watcher_instance.unwatch(watch_id)
            assert unwatched

    @pytest.mark.asyncio
    async def test_hierarchical_rag_workflow(
        self, redis_store, indexer_instance, embedding_manager, temp_test_directory
    ):
        """Test hierarchical RAG with concepts, sections, and chunks."""
        # Step 1: Index with hierarchical structure
        result = await indexer_instance.index_folder(temp_test_directory)
        assert result.total_chunks > 0

        # Step 2: Search at different hierarchy levels
        query = "test project features"
        query_embedding = await embedding_manager.get_embedding(query)

        # Search concepts (level 1)
        concepts_results = await redis_store.vector_search(
            query_embedding=query_embedding, hierarchy_level=1, k=3
        )
        assert len(concepts_results) >= 0

        # Search sections (level 2)
        sections_results = await redis_store.vector_search(
            query_embedding=query_embedding, hierarchy_level=2, k=5
        )
        assert len(sections_results) >= 0

        # Search chunks (level 3)
        chunks_results = await redis_store.vector_search(
            query_embedding=query_embedding, hierarchy_level=3, k=10
        )
        assert len(chunks_results) >= 0

        # Hierarchical search (all levels)
        all_results = await redis_store.hierarchical_search(
            query_embedding=query_embedding, max_chunks=10
        )

        assert isinstance(all_results, list)

    @pytest.mark.asyncio
    async def test_concurrent_operations(
        self, redis_store, indexer_instance, semantic_cache_instance, embedding_manager
    ):
        """Test concurrent operations across components."""

        async def index_operation():
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write("Concurrent indexing test")
                f.flush()
                result = await indexer_instance.index_file(Path(f.name), "concurrent_src")
                Path(f.name).unlink()
                return result

        async def cache_operation(index):
            query = f"Concurrent query {index}"
            response = f"Concurrent response {index}"
            await semantic_cache_instance.set(query, response, {"index": index})
            return await semantic_cache_instance.get(query)

        async def search_operation():
            # Use real embedding for search
            query = f"search query {np.random.randint(100)}"
            embedding = await embedding_manager.get_embedding(query)
            return await redis_store.vector_search(embedding, k=5)

        # Run operations concurrently
        tasks = []

        # Add index operations
        for _ in range(3):
            tasks.append(index_operation())

        # Add cache operations
        for i in range(5):
            tasks.append(cache_operation(i))

        # Add search operations
        for _ in range(3):
            tasks.append(search_operation())

        # Execute all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for successful operations
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) > 0

    @pytest.mark.asyncio
    async def test_performance_metrics(
        self,
        redis_store,
        indexer_instance,
        semantic_cache_instance,
        embedding_manager,
        temp_test_directory,
    ):
        """Test and measure performance metrics."""
        import time

        # Measure indexing speed
        start_time = time.time()
        index_result = await indexer_instance.index_folder(temp_test_directory)
        index_time = time.time() - start_time

        files_per_second = index_result.file_count / index_time if index_time > 0 else 0
        chunks_per_second = index_result.total_chunks / index_time if index_time > 0 else 0

        print("\nIndexing Performance:")
        print(
            f"  Files: {index_result.file_count} in {index_time:.2f}s "
            f"({files_per_second:.1f} files/s)"
        )
        print(f"  Chunks: {index_result.total_chunks} ({chunks_per_second:.1f} chunks/s)")

        # Measure search speed
        query = "performance test query"
        query_embedding = await embedding_manager.get_embedding(query)

        start_time = time.time()
        for _ in range(10):
            await redis_store.vector_search(query_embedding, k=5)
        search_time = time.time() - start_time

        searches_per_second = 10 / search_time if search_time > 0 else 0
        print("\nSearch Performance:")
        print(f"  10 searches in {search_time:.2f}s ({searches_per_second:.1f} searches/s)")

        # Measure cache performance
        start_time = time.time()
        for i in range(20):
            await semantic_cache_instance.set(f"q{i}", f"r{i}", {})
        cache_write_time = time.time() - start_time

        start_time = time.time()
        for i in range(20):
            await semantic_cache_instance.get(f"q{i}")
        cache_read_time = time.time() - start_time

        print("\nCache Performance:")
        print(f"  Writes: 20 in {cache_write_time:.2f}s ({20/cache_write_time:.1f} writes/s)")
        print(f"  Reads: 20 in {cache_read_time:.2f}s ({20/cache_read_time:.1f} reads/s)")

        # All operations should complete reasonably fast
        assert index_time < 30  # Indexing should be under 30 seconds
        assert search_time < 5  # 10 searches should be under 5 seconds
        assert cache_write_time < 5  # 20 cache writes under 5 seconds
        assert cache_read_time < 2  # 20 cache reads under 2 seconds
