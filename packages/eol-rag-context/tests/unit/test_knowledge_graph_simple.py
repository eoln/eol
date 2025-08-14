"""
Simple tests for knowledge_graph to ensure basic functionality.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

# Mock dependencies
sys.modules["networkx"] = MagicMock()

from eol.rag_context.knowledge_graph import (
    Entity,
    EntityType,
    KnowledgeGraphBuilder,
    Relationship,
    RelationType,
)


class TestKnowledgeGraphSimple:
    """Simple tests for knowledge graph."""

    @pytest.fixture
    def graph_builder(self):
        """Create knowledge graph builder with mocked components."""
        mock_redis = AsyncMock()
        mock_redis.redis = MagicMock()
        mock_redis.redis.hset = MagicMock()
        mock_redis.redis.hget = MagicMock()
        mock_redis.redis.scan = MagicMock(return_value=(0, []))

        mock_embeddings = AsyncMock()
        mock_embeddings.get_embedding = AsyncMock(
            return_value=np.random.rand(384).astype(np.float32)
        )

        builder = KnowledgeGraphBuilder(
            redis_store=mock_redis, embedding_manager=mock_embeddings
        )
        return builder

    def test_builder_initialization(self, graph_builder):
        """Test that builder initializes correctly."""
        assert graph_builder is not None
        assert graph_builder.graph is not None
        assert graph_builder.entities == {}
        assert graph_builder.relationships == []

    def test_entity_dataclass(self):
        """Test Entity dataclass."""
        entity = Entity(
            id="test_id",
            name="Test Entity",
            type=EntityType.CONCEPT,
            content="Test content",
            properties={"key": "value"},
            source_ids={"source1"},
        )

        assert entity.id == "test_id"
        assert entity.name == "Test Entity"
        assert entity.type == EntityType.CONCEPT
        assert entity.content == "Test content"
        assert entity.properties["key"] == "value"
        assert "source1" in entity.source_ids

    def test_relationship_dataclass(self):
        """Test Relationship dataclass."""
        rel = Relationship(
            source_id="source",
            target_id="target",
            type=RelationType.RELATES_TO,
            properties={"strength": "high"},
            weight=0.8,
        )

        assert rel.source_id == "source"
        assert rel.target_id == "target"
        assert rel.type == RelationType.RELATES_TO
        assert rel.properties["strength"] == "high"
        assert rel.weight == 0.8

    @pytest.mark.asyncio
    async def test_build_from_documents(self, graph_builder):
        """Test build_from_documents method."""
        # Mock Redis to return no documents
        graph_builder.redis.redis.scan.return_value = (0, [])

        # Should not crash
        await graph_builder.build_from_documents(source_id="test", max_documents=10)

        assert graph_builder.graph is not None

    @pytest.mark.asyncio
    async def test_query_subgraph(self, graph_builder):
        """Test query_subgraph method."""
        # Mock embedding
        graph_builder.embeddings.get_embedding.return_value = np.random.rand(
            384
        ).astype(np.float32)

        # Mock Redis search
        graph_builder.redis.search_similar = AsyncMock(return_value=[])

        # Query subgraph
        result = await graph_builder.query_subgraph(
            query="test query", max_depth=2, max_entities=10
        )

        assert result is not None
        assert hasattr(result, "entities")
        assert hasattr(result, "relationships")

    @pytest.mark.asyncio
    async def test_discover_patterns(self, graph_builder):
        """Test discover_patterns method."""
        # Should return empty patterns for empty graph
        patterns = await graph_builder.discover_patterns(min_support=0.1)

        assert isinstance(patterns, list)

    def test_entity_types(self):
        """Test all entity types."""
        for entity_type in EntityType:
            entity = Entity(
                id=f"test_{entity_type.value}",
                name=f"Test {entity_type.value}",
                type=entity_type,
            )
            assert entity.type == entity_type

    def test_relationship_types(self):
        """Test all relationship types."""
        for rel_type in RelationType:
            rel = Relationship(source_id="source", target_id="target", type=rel_type)
            assert rel.type == rel_type

    @pytest.mark.asyncio
    async def test_store_graph(self, graph_builder):
        """Test _store_graph method."""
        # Should not crash with empty graph
        await graph_builder._store_graph()

        # Redis methods might not be called with empty graph
        assert graph_builder.redis is not None

    def test_graph_attributes(self, graph_builder):
        """Test graph attributes."""
        assert hasattr(graph_builder, "graph")
        assert hasattr(graph_builder, "entities")
        assert hasattr(graph_builder, "relationships")
        assert hasattr(graph_builder, "redis")
        assert hasattr(graph_builder, "embeddings")


if __name__ == "__main__":
    print("✅ Knowledge graph simple tests ready!")
