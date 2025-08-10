"""
Integration tests requiring Redis.
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path

from eol.rag_context.redis_client import RedisVectorStore, VectorDocument
from eol.rag_context.semantic_cache import SemanticCache
from eol.rag_context.knowledge_graph import KnowledgeGraphBuilder, EntityType, RelationType
from eol.rag_context.file_watcher import FileWatcher, ChangeType


@pytest.mark.redis
class TestRedisIntegration:
    """Integration tests with Redis."""
    
    @pytest.mark.asyncio
    async def test_redis_connection(self, redis_store):
        """Test Redis connection."""
        # Ping Redis
        pong = await redis_store.async_redis.ping()
        assert pong
    
    @pytest.mark.asyncio
    async def test_hierarchical_indexes(self, redis_store, test_config):
        """Test creating hierarchical indexes."""
        redis_store.create_hierarchical_indexes(test_config.embedding.dimension)
        
        # Verify indexes exist
        try:
            await redis_store.redis.ft(f"{test_config.index.index_name}_concept").info()
            await redis_store.redis.ft(f"{test_config.index.index_name}_section").info()
            await redis_store.redis.ft(f"{test_config.index.index_name}_chunk").info()
        except Exception as e:
            pytest.fail(f"Failed to create indexes: {e}")
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_document(self, redis_store, test_config):
        """Test storing and retrieving documents."""
        # Create test document
        doc = VectorDocument(
            id="test_doc_1",
            content="This is test content",
            embedding=np.random.randn(test_config.embedding.dimension).astype(np.float32),
            metadata={"test": "metadata"},
            hierarchy_level=3
        )
        
        # Store document
        await redis_store.store_document(doc)
        
        # Retrieve using vector search
        results = await redis_store.vector_search(
            doc.embedding,
            hierarchy_level=3,
            k=1
        )
        
        assert len(results) > 0
        assert results[0][0] == "test_doc_1"
        assert results[0][2]["content"] == "This is test content"
    
    @pytest.mark.asyncio
    async def test_hierarchical_search(self, redis_store, test_config):
        """Test hierarchical search."""
        # Create documents at different levels
        concept = VectorDocument(
            id="concept_1",
            content="High-level concept",
            embedding=np.random.randn(test_config.embedding.dimension).astype(np.float32),
            hierarchy_level=1,
            children_ids=["section_1"]
        )
        
        section = VectorDocument(
            id="section_1",
            content="Section content",
            embedding=np.random.randn(test_config.embedding.dimension).astype(np.float32),
            hierarchy_level=2,
            parent_id="concept_1",
            children_ids=["chunk_1"]
        )
        
        chunk = VectorDocument(
            id="chunk_1",
            content="Detailed chunk",
            embedding=np.random.randn(test_config.embedding.dimension).astype(np.float32),
            hierarchy_level=3,
            parent_id="section_1"
        )
        
        # Store all documents
        await redis_store.store_document(concept)
        await redis_store.store_document(section)
        await redis_store.store_document(chunk)
        
        # Perform hierarchical search
        query_embedding = np.random.randn(test_config.embedding.dimension).astype(np.float32)
        results = await redis_store.hierarchical_search(query_embedding, max_chunks=10)
        
        assert len(results) > 0
        assert any(r["id"] in ["concept_1", "section_1", "chunk_1"] for r in results)


@pytest.mark.redis
class TestSemanticCacheIntegration:
    """Test semantic cache with Redis."""
    
    @pytest.mark.asyncio
    async def test_cache_initialization(self, redis_store, mock_embedding_manager, test_config):
        """Test cache initialization."""
        cache = SemanticCache(
            test_config.cache,
            mock_embedding_manager,
            redis_store
        )
        
        await cache.initialize()
        
        # Cache index should be created
        try:
            await redis_store.redis.ft("cache_index").info()
        except Exception as e:
            pytest.fail(f"Cache index not created: {e}")
    
    @pytest.mark.asyncio
    async def test_cache_get_and_set(self, redis_store, mock_embedding_manager, test_config):
        """Test cache get and set operations."""
        test_config.cache.enabled = True
        cache = SemanticCache(
            test_config.cache,
            mock_embedding_manager,
            redis_store
        )
        await cache.initialize()
        
        # Set cache entry
        query = "What is machine learning?"
        response = "Machine learning is a subset of AI..."
        await cache.set(query, response)
        
        # Get from cache (exact match)
        cached = await cache.get(query)
        assert cached == response
        
        # Stats should show hit
        stats = cache.get_stats()
        assert stats["hits"] == 1
    
    @pytest.mark.asyncio
    async def test_cache_similarity_threshold(self, redis_store, mock_embedding_manager, test_config):
        """Test cache similarity threshold."""
        test_config.cache.enabled = True
        test_config.cache.similarity_threshold = 0.95
        
        cache = SemanticCache(
            test_config.cache,
            mock_embedding_manager,
            redis_store
        )
        await cache.initialize()
        
        # Set cache entry
        await cache.set("What is Python?", "Python is a programming language")
        
        # Similar query (but not exact)
        # With mock embeddings, similarity is random
        cached = await cache.get("What's Python?")
        
        # Result depends on random similarity
        stats = cache.get_stats()
        assert stats["queries"] == 1


@pytest.mark.redis
class TestKnowledgeGraphIntegration:
    """Test knowledge graph with Redis."""
    
    @pytest.mark.asyncio
    async def test_build_knowledge_graph(self, redis_store, mock_embedding_manager, indexed_documents):
        """Test building knowledge graph from documents."""
        graph_builder = KnowledgeGraphBuilder(redis_store, mock_embedding_manager)
        
        await graph_builder.build_from_documents()
        
        stats = graph_builder.get_graph_stats()
        assert stats["entity_count"] > 0
        assert stats["relationship_count"] >= 0
    
    @pytest.mark.asyncio
    async def test_query_subgraph(self, redis_store, mock_embedding_manager):
        """Test querying knowledge subgraph."""
        graph_builder = KnowledgeGraphBuilder(redis_store, mock_embedding_manager)
        
        # Add some test entities
        from eol.rag_context.knowledge_graph import Entity
        
        entity1 = Entity(
            id="entity_1",
            name="Test Entity 1",
            type=EntityType.CONCEPT,
            content="Test content 1",
            embedding=np.random.randn(128).astype(np.float32)
        )
        
        entity2 = Entity(
            id="entity_2",
            name="Test Entity 2",
            type=EntityType.FUNCTION,
            content="Test content 2",
            embedding=np.random.randn(128).astype(np.float32)
        )
        
        graph_builder.entities = {"entity_1": entity1, "entity_2": entity2}
        await graph_builder._store_graph()
        
        # Query subgraph
        subgraph = await graph_builder.query_subgraph("test query", max_depth=1)
        
        assert len(subgraph.entities) > 0
        assert subgraph.metadata["query"] == "test query"


@pytest.mark.redis
class TestFileWatcherIntegration:
    """Test file watcher with real file system."""
    
    @pytest.mark.asyncio
    async def test_watch_directory(self, redis_store, mock_embedding_manager, test_config, temp_dir):
        """Test watching directory for changes."""
        from eol.rag_context.document_processor import DocumentProcessor
        
        processor = DocumentProcessor(test_config.document, test_config.chunking)
        indexer = DocumentIndexer(
            test_config,
            processor,
            mock_embedding_manager,
            redis_store
        )
        
        watcher = FileWatcher(indexer, use_polling=True)  # Use polling for tests
        await watcher.start()
        
        try:
            # Start watching
            source_id = await watcher.watch(temp_dir, recursive=False)
            assert source_id is not None
            
            # Create a new file
            test_file = temp_dir / "new_file.txt"
            test_file.write_text("New content")
            
            # Wait for processing
            await asyncio.sleep(3)
            
            # Check stats
            stats = watcher.get_stats()
            assert stats["watched_sources"] == 1
            
        finally:
            await watcher.stop()
    
    @pytest.mark.asyncio
    async def test_file_change_detection(self, redis_store, mock_embedding_manager, test_config, temp_dir):
        """Test detecting file changes."""
        from eol.rag_context.document_processor import DocumentProcessor
        from eol.rag_context.indexer import DocumentIndexer
        
        processor = DocumentProcessor(test_config.document, test_config.chunking)
        indexer = DocumentIndexer(
            test_config,
            processor,
            mock_embedding_manager,
            redis_store
        )
        
        watcher = FileWatcher(indexer, use_polling=True, debounce_seconds=1.0)
        
        # Track changes
        changes_detected = []
        
        def track_change(change):
            changes_detected.append(change)
        
        watcher.add_change_callback(track_change)
        
        await watcher.start()
        
        try:
            # Create initial file
            test_file = temp_dir / "test.txt"
            test_file.write_text("Initial content")
            
            # Start watching
            await watcher.watch(temp_dir)
            
            # Modify file
            test_file.write_text("Modified content")
            
            # Wait for detection
            await asyncio.sleep(2)
            
            # Should detect modification
            assert len(changes_detected) > 0
            assert any(c.change_type == ChangeType.MODIFIED for c in changes_detected)
            
        finally:
            await watcher.stop()


@pytest.mark.redis
class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_indexing_pipeline(
        self,
        redis_store,
        mock_embedding_manager,
        test_config,
        sample_documents
    ):
        """Test complete indexing pipeline."""
        from eol.rag_context.document_processor import DocumentProcessor
        from eol.rag_context.indexer import DocumentIndexer
        
        # Create components
        processor = DocumentProcessor(test_config.document, test_config.chunking)
        indexer = DocumentIndexer(
            test_config,
            processor,
            mock_embedding_manager,
            redis_store
        )
        
        # Index folder
        folder = sample_documents["markdown"].parent
        result = await indexer.index_folder(folder)
        
        assert result.file_count == len(sample_documents)
        assert result.total_chunks > 0
        
        # Build knowledge graph
        graph = KnowledgeGraphBuilder(redis_store, mock_embedding_manager)
        await graph.build_from_documents(result.source_id)
        
        graph_stats = graph.get_graph_stats()
        assert graph_stats["entity_count"] > 0
        
        # Test search
        query_embedding = np.random.randn(test_config.embedding.dimension).astype(np.float32)
        search_results = await redis_store.hierarchical_search(query_embedding)
        
        assert len(search_results) > 0
    
    @pytest.mark.asyncio
    async def test_cache_with_indexing(
        self,
        redis_store,
        mock_embedding_manager,
        test_config,
        indexed_documents
    ):
        """Test semantic cache with indexed documents."""
        test_config.cache.enabled = True
        
        cache = SemanticCache(
            test_config.cache,
            mock_embedding_manager,
            redis_store
        )
        await cache.initialize()
        
        # Cache some queries
        queries = [
            ("What is Python?", "Python is a programming language"),
            ("How to use Redis?", "Redis is an in-memory database"),
            ("What is RAG?", "RAG is Retrieval Augmented Generation")
        ]
        
        for query, response in queries:
            await cache.set(query, response)
        
        # Test retrieval
        for query, expected_response in queries:
            cached = await cache.get(query)
            assert cached == expected_response
        
        # Check cache stats
        stats = cache.get_stats()
        assert stats["hits"] == len(queries)
        
        # Test cache optimization
        report = await cache.optimize()
        assert "recommendations" in report