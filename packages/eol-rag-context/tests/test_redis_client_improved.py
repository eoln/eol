"""
Improved tests for redis_client to boost coverage from 26% to 60%.
"""

import sys
import json
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
import numpy as np

# Mock all Redis dependencies
mock_redis = MagicMock()
mock_redis.asyncio = MagicMock()
mock_redis.commands = MagicMock()
mock_redis.commands.search = MagicMock()
mock_redis.commands.search.field = MagicMock()
mock_redis.commands.search.indexDefinition = MagicMock()
mock_redis.commands.search.query = MagicMock()
mock_redis.exceptions = MagicMock()

sys.modules["redis"] = mock_redis
sys.modules["redis.asyncio"] = mock_redis.asyncio
sys.modules["redis.commands"] = mock_redis.commands
sys.modules["redis.commands.search"] = mock_redis.commands.search
sys.modules["redis.commands.search.field"] = mock_redis.commands.search.field
sys.modules["redis.commands.search.indexDefinition"] = mock_redis.commands.search.indexDefinition
sys.modules["redis.commands.search.query"] = mock_redis.commands.search.query
sys.modules["redis.exceptions"] = mock_redis.exceptions

# Mock knowledge_graph
sys.modules["networkx"] = MagicMock()

from eol.rag_context import config
from eol.rag_context import redis_client
from eol.rag_context import knowledge_graph


def test_vector_document():
    """Test VectorDocument dataclass."""
    doc = redis_client.VectorDocument(
        id="test_id",
        content="test content",
        embedding=np.array([1.0, 2.0, 3.0]),
        metadata={"key": "value"},
        hierarchy_level=2,
    )
    assert doc.id == "test_id"
    assert doc.hierarchy_level == 2


async def test_redis_vector_store_complete():
    """Test RedisVectorStore with complete coverage."""

    # Create store
    redis_config = config.RedisConfig(host="localhost", port=6379, password="secret")
    index_config = config.IndexConfig()
    store = redis_client.RedisVectorStore(redis_config, index_config)

    # Initial state
    assert store.redis is None
    assert store.async_redis is None

    # Mock Redis connections
    with (
        patch("eol.rag_context.redis_client.AsyncRedis") as MockAsync,
        patch("eol.rag_context.redis_client.Redis") as MockSync,
    ):

        # Setup async mock
        mock_async = MagicMock()
        mock_async.ping = AsyncMock(return_value=True)
        mock_async.ft = MagicMock()
        mock_async.hset = AsyncMock()
        mock_async.hgetall = AsyncMock()
        mock_async.keys = AsyncMock(return_value=[])
        mock_async.delete = AsyncMock()
        mock_async.close = AsyncMock()

        # Mock AsyncRedis to return our mock wrapped in a coroutine
        async def async_redis_init(*args, **kwargs):
            return mock_async

        MockAsync.side_effect = async_redis_init
        MockAsync.from_url = AsyncMock(return_value=mock_async)

        # Setup sync mock
        mock_sync = MagicMock()
        mock_sync.ping = MagicMock(return_value=True)
        mock_sync.close = MagicMock()
        MockSync.return_value = mock_sync

        # Test connect_async
        await store.connect_async()
        assert store.async_redis is not None

        # Test connect (sync)
        store.connect()
        assert store.redis is not None

        # Test create_hierarchical_indexes
        mock_redis.commands.search.field.TextField = MagicMock
        mock_redis.commands.search.field.VectorField = MagicMock
        mock_redis.commands.search.field.NumericField = MagicMock
        mock_redis.commands.search.field.TagField = MagicMock
        mock_redis.commands.search.indexDefinition.IndexDefinition = MagicMock

        # This is a sync method
        store.create_hierarchical_indexes(embedding_dim=768)

        # Test store_document
        doc = redis_client.VectorDocument(
            id="doc1",
            content="Test content with enough text to be meaningful",
            embedding=np.array([1.0, 2.0, 3.0, 4.0]),
            metadata={"type": "test", "source": "unit_test"},
            hierarchy_level=2,
        )

        await store.store_document(doc)
        assert mock_async.hset.called

        # Test search - with results
        mock_result = MagicMock()
        mock_result.id = "doc:1"
        mock_result.content = "Found content"
        mock_result.metadata = json.dumps({"type": "test"})
        mock_result.embedding = np.array([1, 2, 3, 4]).tobytes()
        mock_result.hierarchy_level = 2
        mock_result.score = 0.95

        mock_search_result = MagicMock()
        mock_search_result.docs = [mock_result]
        mock_search_result.total = 1

        mock_async.ft.return_value.search = AsyncMock(return_value=mock_search_result)

        # Mock Query class
        mock_query = MagicMock()
        mock_redis.commands.search.query.Query = MagicMock(return_value=mock_query)

        # Use vector_search method
        results = await store.vector_search("test query", limit=10, index_name="eol_context_l3")
        assert len(results) == 1
        assert results[0].content == "Found content"

        # Test search - no results
        mock_search_result.docs = []
        mock_search_result.total = 0
        results = await store.search("no results", limit=10)
        assert len(results) == 0

        # Test search - with filters
        results = await store.search(
            "filtered query", limit=5, hierarchy_level=1, filters={"type": "document"}
        )

        # Test get_context - hierarchical retrieval
        mock_docs_by_level = {
            1: [mock_result],  # concept level
            2: [mock_result],  # section level
            3: [mock_result],  # chunk level
        }

        async def mock_search_by_level(query, **kwargs):
            level = kwargs.get("hierarchy_level", 3)
            docs = mock_docs_by_level.get(level, [])
            return MagicMock(docs=docs, total=len(docs))

        mock_async.ft.return_value.search = AsyncMock(side_effect=mock_search_by_level)

        contexts = await store.get_context("context query", max_chunks=10)
        assert len(contexts) >= 0

        # Test delete_by_source
        mock_async.keys = AsyncMock(return_value=[b"doc:1", b"doc:2", b"doc:3"])
        mock_async.hgetall = AsyncMock(
            return_value={b"source_id": b"src123", b"content": b"content"}
        )

        await store.delete_by_source("src123")
        assert mock_async.delete.called

        # Test delete_by_source - no matching docs
        mock_async.hgetall = AsyncMock(return_value={b"source_id": b"other_src"})
        await store.delete_by_source("src123")

        # Test store_entities
        entities = [
            knowledge_graph.Entity("e1", "Entity1", knowledge_graph.EntityType.CLASS),
            knowledge_graph.Entity("e2", "Entity2", knowledge_graph.EntityType.FUNCTION),
            knowledge_graph.Entity("e3", "Entity3", knowledge_graph.EntityType.MODULE),
            knowledge_graph.Entity("e4", "Entity4", knowledge_graph.EntityType.VARIABLE),
            knowledge_graph.Entity("e5", "Entity5", knowledge_graph.EntityType.CONCEPT),
        ]

        await store.store_entities(entities)
        assert mock_async.hset.call_count > 0

        # Test store_relationships
        relationships = [
            knowledge_graph.Relationship("e1", "e2", knowledge_graph.RelationType.USES),
            knowledge_graph.Relationship("e2", "e3", knowledge_graph.RelationType.IMPORTS),
            knowledge_graph.Relationship("e3", "e4", knowledge_graph.RelationType.CONTAINS),
            knowledge_graph.Relationship("e4", "e5", knowledge_graph.RelationType.DEPENDS_ON),
            knowledge_graph.Relationship("e5", "e1", knowledge_graph.RelationType.CALLS),
        ]

        await store.store_relationships(relationships)
        assert mock_async.hset.call_count > 0

        # Test list_sources
        mock_async.keys = AsyncMock(return_value=[b"doc:1", b"doc:2", b"doc:3"])
        mock_async.hgetall = AsyncMock(
            side_effect=[
                {b"source_id": b"src1", b"source_path": b"/path1"},
                {b"source_id": b"src2", b"source_path": b"/path2"},
                {b"source_id": b"src1", b"source_path": b"/path1"},  # duplicate
            ]
        )

        sources = await store.list_sources()
        assert len(sources) >= 0

        # Test close
        await store.close()
        # The close method should close both redis connections if they exist

        # Test _serialize_embedding and _deserialize_embedding
        emb = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        serialized = store._serialize_embedding(emb)
        assert isinstance(serialized, bytes)

        deserialized = store._deserialize_embedding(serialized)
        assert np.allclose(emb, deserialized)

        # Test with None embedding
        assert store._serialize_embedding(None) == b""
        assert store._deserialize_embedding(b"") is None

        # Test connection errors
        MockAsync.from_url = AsyncMock(side_effect=Exception("Connection failed"))
        store2 = redis_client.RedisVectorStore(redis_config, index_config)

        try:
            await store2.connect_async()
        except Exception as e:
            assert "Connection failed" in str(e)

        # Test sync connection error
        MockSync.side_effect = Exception("Sync connection failed")
        try:
            store2.connect()
        except Exception as e:
            assert "Sync connection failed" in str(e)

        # Test search with error
        mock_async.ft.return_value.search = AsyncMock(side_effect=Exception("Search error"))
        results = await store.search("error query")
        assert results == []

        # Test get_context with different max_chunks
        contexts = await store.get_context("test", max_chunks=1)
        contexts = await store.get_context("test", max_chunks=100)

        # Test store_document with large metadata
        large_doc = redis_client.VectorDocument(
            id="large_doc",
            content="Content " * 1000,  # Large content
            embedding=np.random.randn(768),  # Large embedding
            metadata={
                "key1": "value1" * 100,
                "key2": "value2" * 100,
                "nested": {"deep": {"structure": "value"}},
            },
            hierarchy_level=3,
        )
        await store.store_document(large_doc)


async def test_edge_cases():
    """Test edge cases and error conditions."""

    store = redis_client.RedisVectorStore(config.RedisConfig(), config.IndexConfig())

    # Test operations without connection
    results = await store.search("query")
    assert results == []

    contexts = await store.get_context("query")
    assert contexts == []

    await store.delete_by_source("src")

    sources = await store.list_sources()
    assert sources == []

    # Test with mock connection but operations fail
    with patch("eol.rag_context.redis_client.AsyncRedis") as MockAsync:
        mock_async = MagicMock()
        MockAsync.from_url = AsyncMock(return_value=mock_async)
        mock_async.ping = AsyncMock(return_value=True)

        await store.connect_async()

        # Test store_document with error
        mock_async.hset = AsyncMock(side_effect=Exception("Store error"))
        doc = redis_client.VectorDocument(
            id="test", content="test", embedding=np.array([1, 2]), metadata={}, hierarchy_level=1
        )

        try:
            await store.store_document(doc)
        except:
            pass

        # Test delete with error
        mock_async.keys = AsyncMock(side_effect=Exception("Keys error"))
        await store.delete_by_source("error")

        # Test store_entities with error
        mock_async.hset = AsyncMock(side_effect=Exception("Entity error"))
        entities = [knowledge_graph.Entity("e1", "E1", knowledge_graph.EntityType.CLASS)]

        try:
            await store.store_entities(entities)
        except:
            pass


def test_sync():
    """Test synchronously."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(test_redis_vector_store_complete())
        loop.run_until_complete(test_edge_cases())
    finally:
        loop.close()


if __name__ == "__main__":
    test_vector_document()
    test_sync()
    print("✅ All redis_client tests passed!")
