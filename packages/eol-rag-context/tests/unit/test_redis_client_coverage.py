"""Unit tests to increase redis_client.py coverage to 80%."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from eol.rag_context.config import EmbeddingConfig, IndexConfig, RAGConfig, RedisConfig
from eol.rag_context.redis_client import RedisVectorStore, VectorDocument


@pytest.fixture
def redis_config():
    """Create test Redis configuration."""
    return RedisConfig(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=False,
    )


@pytest.fixture
def index_config():
    """Create test index configuration."""
    return IndexConfig(
        m=16,
        ef_construction=200,
        ef_runtime=10,
        concept_vectorset="test_concept",
        section_vectorset="test_section",
        chunk_vectorset="test_chunk",
    )


@pytest.fixture
def embedding_config():
    """Create test embedding configuration."""
    return EmbeddingConfig(
        provider="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        dimension=384,
    )


@pytest.fixture
def rag_config(redis_config, index_config, embedding_config):
    """Create test RAG configuration."""
    config = RAGConfig()
    config.redis = redis_config
    config.index = index_config
    config.embedding = embedding_config
    return config


@pytest.fixture
def vector_store(redis_config, index_config):
    """Create RedisVectorStore instance with mocked Redis clients."""
    store = RedisVectorStore(redis_config, index_config)
    # Mock both sync and async Redis clients
    store.redis = MagicMock()
    store.async_redis = AsyncMock()
    return store


class TestRedisVectorStoreVectorSearch:
    """Test vector_search method for coverage."""

    @pytest.mark.asyncio
    async def test_vector_search_success(self, vector_store):
        """Test successful vector search with results."""
        # Setup
        query_embedding = np.random.rand(384).astype(np.float32)

        # Mock VSIM command response (alternating element_id, score pairs)
        mock_results = ["doc_1", "0.95", "doc_2", "0.89", "doc_3", "0.82"]
        vector_store.async_redis.execute_command = AsyncMock(return_value=mock_results)

        # Mock hash retrieval for documents (using async_redis, not sync redis)
        vector_store.async_redis.hgetall = AsyncMock(
            side_effect=[
                {
                    b"content": b"Document 1 content",
                    b"metadata": json.dumps({"source": "file1.txt"}).encode(),
                },
                {
                    b"content": b"Document 2 content",
                    b"metadata": json.dumps({"source": "file2.txt"}).encode(),
                },
                {
                    b"content": b"Document 3 content",
                    b"metadata": json.dumps({"source": "file3.txt"}).encode(),
                },
            ]
        )

        # Execute
        results = await vector_store.vector_search(query_embedding, k=3, hierarchy_level=3)

        # Verify
        assert len(results) == 3
        assert results[0][0] == "doc_1"  # document_id
        assert results[0][1] == 0.95  # score
        assert "content" in results[0][2]  # document_data

        # Verify VSIM command was called with correct args
        call_args = vector_store.async_redis.execute_command.call_args[0]
        assert call_args[0] == "VSIM"
        assert call_args[1] == "test_chunk"  # chunk vectorset for level 3
        assert call_args[2] == "VALUES"
        assert call_args[3] == str(len(query_embedding.flatten()))  # embedding dimension

    @pytest.mark.asyncio
    async def test_vector_search_concept_level(self, vector_store):
        """Test vector search at concept level (level 1)."""
        query_embedding = np.random.rand(384).astype(np.float32)

        # Mock VSIM response
        vector_store.async_redis.execute_command = AsyncMock(return_value=["doc_1", "0.9"])
        vector_store.async_redis.hgetall = AsyncMock(
            return_value={b"content": b"Concept content", b"metadata": json.dumps({}).encode()}
        )

        # Execute search at concept level
        await vector_store.vector_search(query_embedding, k=1, hierarchy_level=1)

        # Verify concept vectorset was used
        call_args = vector_store.async_redis.execute_command.call_args[0]
        assert call_args[1] == "test_concept"
        # Check EF parameter for concept level (higher quality)
        assert "EF" in call_args
        ef_index = call_args.index("EF")
        assert int(call_args[ef_index + 1]) == 100  # ef_runtime * 10

    @pytest.mark.asyncio
    async def test_vector_search_section_level(self, vector_store):
        """Test vector search at section level (level 2)."""
        query_embedding = np.random.rand(384).astype(np.float32)

        # Mock VSIM response
        vector_store.async_redis.execute_command = AsyncMock(return_value=["doc_1", "0.9"])
        vector_store.async_redis.hgetall = AsyncMock(
            return_value={b"content": b"Section content", b"metadata": json.dumps({}).encode()}
        )

        # Execute search at section level
        await vector_store.vector_search(query_embedding, k=1, hierarchy_level=2)

        # Verify section vectorset was used
        call_args = vector_store.async_redis.execute_command.call_args[0]
        assert call_args[1] == "test_section"
        # Check EF parameter for section level
        assert "EF" in call_args
        ef_index = call_args.index("EF")
        assert int(call_args[ef_index + 1]) == 50  # ef_runtime * 5

    @pytest.mark.asyncio
    async def test_vector_search_no_results(self, vector_store):
        """Test vector search with no results."""
        query_embedding = np.random.rand(384).astype(np.float32)

        # Mock empty VSIM response
        vector_store.async_redis.execute_command = AsyncMock(return_value=[])

        # Execute
        results = await vector_store.vector_search(query_embedding, k=5)

        # Verify
        assert results == []

    @pytest.mark.asyncio
    async def test_vector_search_vset_not_exists(self, vector_store):
        """Test vector search when vector set doesn't exist."""
        query_embedding = np.random.rand(384).astype(np.float32)

        # Mock VSIM error for non-existent vector set
        vector_store.async_redis.execute_command = AsyncMock(
            side_effect=Exception("VSET does not exist")
        )

        # Execute - should return empty results instead of raising
        results = await vector_store.vector_search(query_embedding, k=5)

        # Verify
        assert results == []

    @pytest.mark.asyncio
    async def test_vector_search_other_error(self, vector_store):
        """Test vector search with other Redis errors."""
        query_embedding = np.random.rand(384).astype(np.float32)

        # Mock other Redis error
        vector_store.async_redis.execute_command = AsyncMock(
            side_effect=Exception("Connection timeout")
        )

        # Execute - should raise the error
        with pytest.raises(Exception, match="Connection timeout"):
            await vector_store.vector_search(query_embedding, k=5)

    @pytest.mark.asyncio
    async def test_vector_search_multidimensional_embedding(self, vector_store):
        """Test vector search with multi-dimensional embedding (should flatten)."""
        # Create 2D embedding that needs flattening
        query_embedding = np.random.rand(1, 384).astype(np.float32)

        # Mock VSIM response
        vector_store.async_redis.execute_command = AsyncMock(return_value=["doc_1", "0.9"])
        vector_store.async_redis.hgetall = AsyncMock(
            return_value={b"content": b"Content", b"metadata": json.dumps({}).encode()}
        )

        # Execute
        await vector_store.vector_search(query_embedding, k=1)

        # Verify embedding was flattened (384 values, not nested)
        call_args = vector_store.async_redis.execute_command.call_args[0]
        assert call_args[3] == "384"
        # Check that values are flat (not nested lists)
        values_start = 4
        assert isinstance(call_args[values_start], str)  # Should be string representation of float

    @pytest.mark.asyncio
    async def test_vector_search_missing_document_data(self, vector_store):
        """Test vector search when document hash is missing."""
        query_embedding = np.random.rand(384).astype(np.float32)

        # Mock VSIM response
        vector_store.async_redis.execute_command = AsyncMock(return_value=["doc_1", "0.9"])

        # Mock missing document hash
        vector_store.async_redis.hgetall = AsyncMock(return_value={})

        # Execute
        results = await vector_store.vector_search(query_embedding, k=1)

        # Verify result still included but with empty content
        assert len(results) == 1
        assert results[0][0] == "doc_1"  # document_id
        assert results[0][2]["content"] == ""  # processed_data["content"]
        assert results[0][2]["metadata"] == {}  # processed_data["metadata"]


class TestRedisVectorStoreHierarchicalSearch:
    """Test hierarchical_search method for coverage."""

    @pytest.mark.asyncio
    async def test_hierarchical_search_adaptive(self, vector_store):
        """Test hierarchical search with adaptive strategy."""
        query_embedding = np.random.rand(384).astype(np.float32)

        # Mock vector_search calls for different levels
        # First call: concept level (level 1)
        concept_results = [
            (
                "concept_1",
                0.95,
                {"content": "Main concept", "metadata": {}, "children": "section_1,section_2"},
            )
        ]
        # Second call: section level (level 2)
        section_results = [
            (
                "section_1",
                0.90,
                {"content": "Section 1", "metadata": {}, "children": "chunk_1,chunk_2"},
            ),
            ("section_2", 0.85, {"content": "Section 2", "metadata": {}, "children": "chunk_3"}),
        ]
        # Third call: chunk level (level 3)
        chunk_results = [
            ("chunk_1", 0.88, {"content": "Chunk 1 content", "metadata": {}}),
            ("chunk_2", 0.82, {"content": "Chunk 2 content", "metadata": {}}),
            ("chunk_3", 0.80, {"content": "Chunk 3 content", "metadata": {}}),
        ]

        vector_store.vector_search = AsyncMock(
            side_effect=[concept_results, section_results, chunk_results]
        )

        # Execute
        results = await vector_store.hierarchical_search(
            query_embedding=query_embedding, max_chunks=5, strategy="adaptive"
        )

        # Verify vector_search was called 3 times (once per level)
        assert vector_store.vector_search.call_count == 3
        # Verify the results structure
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)


class TestRedisVectorStoreConnectionMethods:
    """Test connection-related methods for coverage."""

    @pytest.mark.asyncio
    async def test_connect_async_not_connected(self, vector_store):
        """Test async connection when not already connected."""
        vector_store.async_redis = None

        with patch("redis.asyncio.Redis") as mock_redis_class:
            mock_async_redis = AsyncMock()
            mock_async_redis.ping = AsyncMock(return_value=True)
            mock_redis_class.return_value = mock_async_redis

            # Execute
            await vector_store.connect_async()

            # Verify
            assert vector_store.async_redis is not None
            mock_async_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_async_already_connected(self, vector_store):
        """Test async connection when already connected."""
        # Already has async_redis from fixture
        original_client = vector_store.async_redis

        # Execute
        await vector_store.connect_async()

        # Verify - should use existing connection
        assert vector_store.async_redis is original_client

    @pytest.mark.asyncio
    async def test_close(self, vector_store):
        """Test async close method."""
        # Setup
        vector_store.async_redis = AsyncMock()
        vector_store.async_redis.close = AsyncMock()

        # Execute
        await vector_store.close()

        # Verify
        vector_store.async_redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_not_connected(self, vector_store):
        """Test close when not connected."""
        vector_store.async_redis = None

        # Execute - should not raise
        await vector_store.close()

        # Verify no error occurred
        assert vector_store.async_redis is None


class TestRedisVectorStoreStoreDocument:
    """Test store_document method for coverage."""

    @pytest.mark.asyncio
    async def test_store_document_basic(self, vector_store):
        """Test basic document storage."""
        # Create a test document
        doc = VectorDocument(
            id="test_doc_1",
            content="Test document content",
            embedding=np.random.rand(384).astype(np.float32),
            metadata={"source": "test.txt", "type": "text"},
            hierarchy_level=3,
        )

        # Mock Redis commands
        vector_store.async_redis.hset = AsyncMock(return_value=1)
        vector_store.async_redis.execute_command = AsyncMock(return_value=1)

        # Execute
        await vector_store.store_document(doc)

        # Verify hset was called with document data
        vector_store.async_redis.hset.assert_called_once()
        # Verify VADD was called for vector storage
        vector_store.async_redis.execute_command.assert_called_once()
        call_args = vector_store.async_redis.execute_command.call_args[0]
        assert call_args[0] == "VADD"
        assert call_args[1] == "test_chunk"  # chunk vectorset for level 3

    @pytest.mark.asyncio
    async def test_store_document_with_parent(self, vector_store):
        """Test document storage with parent relationship."""
        # Create document with parent
        doc = VectorDocument(
            id="child_doc",
            content="Child document",
            embedding=np.random.rand(384).astype(np.float32),
            metadata={"source": "child.txt"},
            hierarchy_level=2,
            parent_id="parent_doc",
        )

        # Mock Redis commands
        vector_store.async_redis.hset = AsyncMock(return_value=1)
        vector_store.async_redis.execute_command = AsyncMock(return_value=1)

        # Execute
        await vector_store.store_document(doc)

        # Verify hset includes parent field
        call_args = vector_store.async_redis.hset.call_args
        assert b"parent" in call_args[1][2]  # Check mapping includes parent
        # Verify section vectorset was used for level 2
        vadd_args = vector_store.async_redis.execute_command.call_args[0]
        assert vadd_args[1] == "test_section"


class TestRedisVectorStoreDocumentTree:
    """Test get_document_tree method for coverage."""

    @pytest.mark.asyncio
    async def test_get_document_tree(self, vector_store):
        """Test getting document tree hierarchy."""
        # Mock document data at different levels
        vector_store.async_redis.hgetall = AsyncMock(
            side_effect=[
                # Chunk data
                {
                    b"content": b"Chunk content",
                    b"metadata": json.dumps({"source": "file.txt"}).encode(),
                    b"parent": b"section_1",
                },
                # Section data
                {
                    b"content": b"Section content",
                    b"metadata": json.dumps({}).encode(),
                    b"parent": b"concept_1",
                },
                # Concept data
                {b"content": b"Concept content", b"metadata": json.dumps({}).encode()},
            ]
        )

        # Execute
        result = await vector_store.get_document_tree("chunk_123")

        # Verify structure
        assert "document" in result
        assert "parent" in result
        assert "grandparent" in result
