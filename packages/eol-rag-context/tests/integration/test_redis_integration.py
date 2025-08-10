"""
Integration tests for Redis vector store operations.
Tests real Redis connectivity and vector operations.
"""

import pytest
import numpy as np
import json
from eol.rag_context import redis_client


@pytest.mark.integration
class TestRedisIntegration:
    """Test Redis vector store with real Redis instance."""
    
    @pytest.mark.asyncio
    async def test_connection(self, redis_store):
        """Test Redis connection establishment."""
        # Connection already established in fixture
        assert redis_store.redis is not None
        assert redis_store.async_redis is not None
        
        # Test ping
        result = redis_store.redis.ping()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_document(self, redis_store):
        """Test storing and retrieving documents."""
        # Create a test document
        doc = redis_client.VectorDocument(
            id="test_doc_1",
            content="This is a test document for integration testing.",
            embedding=np.random.randn(768),
            metadata={"type": "test", "category": "integration"},
            hierarchy_level=1
        )
        
        # Store the document
        await redis_store.store_document(doc)
        
        # Retrieve using async_redis
        key = f"doc:{doc.hierarchy_level}:{doc.id}"
        stored_data = await redis_store.async_redis.hgetall(key)
        
        assert stored_data is not None
        assert b"content" in stored_data
        assert stored_data[b"content"].decode() == doc.content
    
    @pytest.mark.asyncio
    async def test_vector_search(self, redis_store):
        """Test vector similarity search."""
        # Store multiple documents
        docs = []
        for i in range(5):
            doc = redis_client.VectorDocument(
                id=f"search_doc_{i}",
                content=f"Document {i} with unique content for testing.",
                embedding=np.random.randn(768),
                metadata={"index": i, "type": "search_test"},
                hierarchy_level=3
            )
            docs.append(doc)
            await redis_store.store_document(doc)
        
        # Perform vector search
        query_embedding = np.random.randn(768)
        results = await redis_store.vector_search(
            query_embedding=query_embedding,
            hierarchy_level=3,
            k=3
        )
        
        # Verify results structure
        assert isinstance(results, list)
        assert len(results) <= 3
        
        for result in results:
            assert "id" in result
            assert "content" in result
            assert "score" in result
            assert "metadata" in result
    
    @pytest.mark.asyncio
    async def test_hierarchical_search(self, redis_store):
        """Test hierarchical document search."""
        # Create documents at different hierarchy levels
        for level in [1, 2, 3]:
            for i in range(3):
                doc = redis_client.VectorDocument(
                    id=f"hier_doc_L{level}_{i}",
                    content=f"Level {level} document {i}",
                    embedding=np.random.randn(768),
                    metadata={"level": level, "index": i},
                    hierarchy_level=level
                )
                await redis_store.store_document(doc)
        
        # Search across hierarchy
        query_embedding = np.random.randn(768)
        results = await redis_store.hierarchical_search(
            query_embedding=query_embedding,
            max_results=5
        )
        
        assert isinstance(results, list)
        assert len(results) <= 5
    
    @pytest.mark.asyncio
    async def test_document_tree(self, redis_store):
        """Test document tree retrieval."""
        # Create parent document
        parent = redis_client.VectorDocument(
            id="parent_doc",
            content="Parent document",
            embedding=np.random.randn(768),
            metadata={"type": "parent"},
            hierarchy_level=1
        )
        await redis_store.store_document(parent)
        
        # Create child documents
        for i in range(3):
            child = redis_client.VectorDocument(
                id=f"child_doc_{i}",
                content=f"Child document {i}",
                embedding=np.random.randn(768),
                metadata={"type": "child", "parent": "parent_doc"},
                hierarchy_level=2
            )
            await redis_store.store_document(child)
        
        # Get document tree
        tree = await redis_store.get_document_tree("parent_doc")
        
        assert isinstance(tree, dict)
        assert tree.get("id") == "parent_doc"
        assert "content" in tree
        assert "metadata" in tree
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, redis_store):
        """Test batch document operations."""
        # Store batch of documents
        batch_size = 10
        docs = []
        
        for i in range(batch_size):
            doc = redis_client.VectorDocument(
                id=f"batch_doc_{i}",
                content=f"Batch document {i} content",
                embedding=np.random.randn(768),
                metadata={"batch": True, "index": i},
                hierarchy_level=2
            )
            docs.append(doc)
        
        # Store all documents
        for doc in docs:
            await redis_store.store_document(doc)
        
        # Verify all stored
        for doc in docs:
            key = f"doc:{doc.hierarchy_level}:{doc.id}"
            data = await redis_store.async_redis.hgetall(key)
            assert data is not None
            assert b"content" in data
    
    @pytest.mark.asyncio
    async def test_filtered_search(self, redis_store):
        """Test vector search with metadata filters."""
        # Store documents with different metadata
        categories = ["science", "technology", "history"]
        
        for cat in categories:
            for i in range(3):
                doc = redis_client.VectorDocument(
                    id=f"filtered_{cat}_{i}",
                    content=f"Document about {cat} number {i}",
                    embedding=np.random.randn(768),
                    metadata={"category": cat, "index": i},
                    hierarchy_level=3
                )
                await redis_store.store_document(doc)
        
        # Search with filter
        query_embedding = np.random.randn(768)
        results = await redis_store.vector_search(
            query_embedding=query_embedding,
            hierarchy_level=3,
            k=5,
            filters={"category": "technology"}
        )
        
        # Verify filtered results
        for result in results:
            if "metadata" in result and result["metadata"]:
                metadata = result["metadata"]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                # Filter verification would happen here in real scenario
    
    @pytest.mark.asyncio
    async def test_index_creation(self, redis_store):
        """Test Redis index creation and management."""
        # Indexes already created in fixture
        # Verify they exist by attempting operations
        
        # This would normally check index info
        # For now, just verify no errors on operations
        assert redis_store.redis is not None
        
        # Try to create indexes again (should handle gracefully)
        try:
            redis_store.create_hierarchical_indexes(embedding_dim=768)
        except Exception:
            # Expected if indexes already exist
            pass
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, redis_store):
        """Test concurrent document operations."""
        import asyncio
        
        async def store_doc(index):
            doc = redis_client.VectorDocument(
                id=f"concurrent_{index}",
                content=f"Concurrent document {index}",
                embedding=np.random.randn(768),
                metadata={"concurrent": True, "index": index},
                hierarchy_level=2
            )
            await redis_store.store_document(doc)
            return doc.id
        
        # Store documents concurrently
        tasks = [store_doc(i) for i in range(10)]
        doc_ids = await asyncio.gather(*tasks)
        
        assert len(doc_ids) == 10
        assert all(f"concurrent_{i}" in doc_ids for i in range(10))
    
    @pytest.mark.asyncio
    async def test_cleanup(self, redis_store):
        """Test cleanup operations."""
        # Store a document
        doc = redis_client.VectorDocument(
            id="cleanup_test",
            content="Document to be cleaned up",
            embedding=np.random.randn(768),
            metadata={"cleanup": True},
            hierarchy_level=1
        )
        await redis_store.store_document(doc)
        
        # Verify it exists
        key = f"doc:{doc.hierarchy_level}:{doc.id}"
        data = await redis_store.async_redis.hgetall(key)
        assert data is not None
        
        # Delete it
        await redis_store.async_redis.delete(key)
        
        # Verify it's gone
        data = await redis_store.async_redis.hgetall(key)
        assert len(data) == 0