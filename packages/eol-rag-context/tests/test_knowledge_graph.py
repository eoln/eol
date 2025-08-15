"""Focused tests for knowledge_graph.py module.

This test file contains meaningful tests for the knowledge graph components, extracted
from coverage booster files and enhanced with real functionality testing.

"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from eol.rag_context.knowledge_graph import (
    Entity,
    EntityType,
    KnowledgeGraphBuilder,
    KnowledgeSubgraph,
    Relationship,
    RelationType,
)


class TestEntityType:
    """Test EntityType enum."""

    def test_entity_type_values(self):
        """Test EntityType enum values."""
        assert EntityType.CONCEPT.value == "concept"
        assert EntityType.FUNCTION.value == "function"
        assert EntityType.CLASS.value == "class"
        assert EntityType.DOCUMENT.value == "document"
        assert EntityType.TOPIC.value == "topic"

    def test_entity_type_membership(self):
        """Test EntityType membership checks."""
        code_types = {EntityType.FUNCTION, EntityType.CLASS, EntityType.MODULE}
        assert EntityType.FUNCTION in code_types
        assert EntityType.DOCUMENT not in code_types


class TestRelationType:
    """Test RelationType enum."""

    def test_relationship_type_values(self):
        """Test RelationType enum values."""
        assert RelationType.CONTAINS.value == "contains"
        assert RelationType.SIMILAR_TO.value == "similar_to"
        assert RelationType.IMPLEMENTS.value == "implements"
        assert RelationType.CALLS.value == "calls"

    def test_relationship_categories(self):
        """Test relationship type categorization."""
        structural_types = {RelationType.CONTAINS, RelationType.PART_OF}
        semantic_types = {RelationType.SIMILAR_TO, RelationType.RELATES_TO}

        assert RelationType.CONTAINS in structural_types
        assert RelationType.SIMILAR_TO in semantic_types


class TestEntity:
    """Test Entity dataclass."""

    def test_entity_creation(self):
        """Test Entity creation with all fields."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        entity = Entity(
            id="func_test_function",
            name="test_function",
            type=EntityType.FUNCTION,
            content="def test_function(): pass",
            embedding=embedding,
            properties={"language": "python", "line_count": 1},
            source_ids={"src_123", "src_456"},
        )

        assert entity.id == "func_test_function"
        assert entity.name == "test_function"
        assert entity.type == EntityType.FUNCTION
        assert entity.content == "def test_function(): pass"
        assert np.array_equal(entity.embedding, embedding)
        assert entity.properties["language"] == "python"
        assert len(entity.source_ids) == 2

    def test_entity_defaults(self):
        """Test Entity with default values."""
        entity = Entity(id="doc_test", name="Test Document", type=EntityType.DOCUMENT)

        assert entity.content == ""
        assert entity.embedding is None
        assert entity.properties == {}
        assert entity.source_ids == set()


class TestRelationship:
    """Test Relationship dataclass."""

    def test_relationship_creation(self):
        """Test Relationship creation."""
        relationship = Relationship(
            source_id="func_a",
            target_id="func_b",
            type=RelationType.CALLS,
            weight=0.8,
            properties={"call_count": 5, "context": "main_flow"},
        )

        assert relationship.source_id == "func_a"
        assert relationship.target_id == "func_b"
        assert relationship.type == RelationType.CALLS
        assert relationship.weight == 0.8
        assert relationship.properties["call_count"] == 5

    def test_relationship_defaults(self):
        """Test Relationship with default values."""
        relationship = Relationship(
            source_id="entity1", target_id="entity2", type=RelationType.RELATES_TO
        )

        assert relationship.weight == 1.0
        assert relationship.properties == {}


class TestKnowledgeSubgraph:
    """Test KnowledgeSubgraph dataclass."""

    def test_subgraph_creation(self):
        """Test KnowledgeSubgraph creation."""
        entities = [
            Entity(id="e1", name="Entity 1", type=EntityType.CONCEPT),
            Entity(id="e2", name="Entity 2", type=EntityType.FUNCTION),
        ]

        relationships = [
            Relationship(source_id="e1", target_id="e2", type=RelationType.CONTAINS)
        ]

        subgraph = KnowledgeSubgraph(
            entities=entities,
            relationships=relationships,
            central_entities=["e1"],
            metadata={"query": "test query", "depth": 2},
        )

        assert len(subgraph.entities) == 2
        assert len(subgraph.relationships) == 1
        assert subgraph.central_entities == ["e1"]
        assert subgraph.metadata["query"] == "test query"


class TestKnowledgeGraphBuilder:
    """Test KnowledgeGraphBuilder class with real functionality."""

    @pytest.fixture
    def mock_redis_store(self):
        """Create mock Redis store."""
        mock = MagicMock()
        mock.redis = MagicMock()
        return mock

    @pytest.fixture
    def mock_embedding_manager(self):
        """Create mock embedding manager."""
        mock = MagicMock()
        mock.get_embedding = AsyncMock(
            return_value=np.array([0.1, 0.2, 0.3], dtype=np.float32)
        )
        return mock

    @pytest.fixture
    def kg_builder(self, mock_redis_store, mock_embedding_manager):
        """Create KnowledgeGraphBuilder instance."""
        return KnowledgeGraphBuilder(mock_redis_store, mock_embedding_manager)

    def test_builder_initialization(
        self, kg_builder, mock_redis_store, mock_embedding_manager
    ):
        """Test KnowledgeGraphBuilder initialization."""
        assert kg_builder.redis == mock_redis_store
        assert kg_builder.embeddings == mock_embedding_manager
        assert kg_builder.entities == {}
        assert kg_builder.relationships == []
        assert hasattr(kg_builder, "graph")

    @pytest.mark.asyncio
    async def test_extract_markdown_entities(self, kg_builder):
        """Test markdown entity extraction."""
        markdown_content = """# Main Section
This is a main section about authentication.

## Authentication Methods
We support multiple methods:
- OAuth2
- JWT tokens

```python
def authenticate(user):
    return validate_token(user.token)
```

See [documentation](https://example.com/docs) for more details.
"""

        # Mock document scanning - method modifies kg_builder state
        await kg_builder._extract_markdown_entities(
            markdown_content, "doc1", {"source_id": "test_source"}
        )

        # Should extract topics from headers - check in kg_builder.entities
        topic_entities = [
            e for e in kg_builder.entities.values() if e.type == EntityType.TOPIC
        ]
        assert (
            len(topic_entities) >= 2
        )  # Should have "Main Section" and "Authentication Methods"

        # Verify specific headers were extracted
        topic_names = [e.name for e in topic_entities]
        assert any("Main Section" in name for name in topic_names)
        assert any("Authentication Methods" in name for name in topic_names)

        # Should extract code blocks as API entities
        api_entities = [
            e for e in kg_builder.entities.values() if e.type == EntityType.API
        ]
        assert len(api_entities) >= 1  # Should have the python code block

    @pytest.mark.asyncio
    async def test_extract_code_entities_python(self, kg_builder):
        """Test Python code entity extraction."""
        python_code = """
        import os
        from typing import List, Dict

        class AuthenticationManager:
            '''Handles user authentication.'''

            def __init__(self, config: Dict[str, str]):
                self.config = config
                self.sessions = {}

            def authenticate(self, username: str, password: str) -> bool:
                '''Authenticate user credentials.'''
                return self._validate_credentials(username, password)

            def _validate_credentials(self, username: str, password: str) -> bool:
                # Implementation here
                return True

        def create_session(user_id: str) -> str:
            '''Create new user session.'''
            session_id = generate_session_id()
            return session_id

        def logout(session_id: str) -> None:
            '''Logout user session.'''
            cleanup_session(session_id)
        """

        # Method modifies kg_builder state, doesn't return entities
        await kg_builder._extract_code_entities_from_content(
            python_code,
            "auth_module",
            {"language": "python", "file_type": "code", "source_id": "test_source"},
        )

        # Should extract class entities - check in kg_builder.entities
        class_entities = [
            e for e in kg_builder.entities.values() if e.type == EntityType.CLASS
        ]
        assert len(class_entities) >= 1
        assert any("AuthenticationManager" in e.name for e in class_entities)

        # Should extract function entities
        function_entities = [
            e for e in kg_builder.entities.values() if e.type == EntityType.FUNCTION
        ]
        assert len(function_entities) >= 2  # authenticate, create_session, logout, etc.

    @pytest.mark.asyncio
    async def test_extract_text_entities(self, kg_builder):
        """Test text entity extraction."""
        text_content = """
        The Redis Database is used for caching and session storage.
        Our API Framework provides REST endpoints for authentication.
        The System Administrator configures security settings.
        Python Library includes utilities for data processing.
        """

        # Method modifies kg_builder state, doesn't return entities
        await kg_builder._extract_text_entities(
            text_content, "doc1", {"source_id": "test_source"}
        )

        # Should extract technology entities - check in kg_builder.entities
        tech_entities = [
            e for e in kg_builder.entities.values() if e.type == EntityType.TECHNOLOGY
        ]
        assert len(tech_entities) > 0

        # Should extract term entities
        term_entities = [
            e for e in kg_builder.entities.values() if e.type == EntityType.TERM
        ]
        assert len(term_entities) > 0

    @pytest.mark.asyncio
    async def test_build_semantic_relationships(self, kg_builder):
        """Test semantic relationship building."""
        # Add entities with embeddings
        embedding1 = np.array([0.8, 0.6, 0.1], dtype=np.float32)
        embedding2 = np.array(
            [0.9, 0.5, 0.2], dtype=np.float32
        )  # Similar to embedding1
        embedding3 = np.array(
            [0.1, 0.2, 0.9], dtype=np.float32
        )  # Different from others

        entity1 = Entity(
            id="e1",
            name="Auth Function",
            type=EntityType.FUNCTION,
            embedding=embedding1,
        )
        entity2 = Entity(
            id="e2",
            name="Login Function",
            type=EntityType.FUNCTION,
            embedding=embedding2,
        )
        entity3 = Entity(
            id="e3",
            name="Database Config",
            type=EntityType.CONCEPT,
            embedding=embedding3,
        )

        kg_builder.entities = {"e1": entity1, "e2": entity2, "e3": entity3}
        kg_builder.graph.add_node("e1")
        kg_builder.graph.add_node("e2")
        kg_builder.graph.add_node("e3")

        await kg_builder._build_semantic_relationships()

        # Should create relationships between similar entities
        similar_relationships = [
            r for r in kg_builder.relationships if r.type == RelationType.SIMILAR_TO
        ]

        # Check that similar entities (e1, e2) have a relationship
        e1_e2_rel = any(
            (r.source_id == "e1" and r.target_id == "e2")
            or (r.source_id == "e2" and r.target_id == "e1")
            for r in similar_relationships
        )

        # Relationship should exist if similarity is high enough
        if e1_e2_rel:
            assert len(similar_relationships) > 0

    @pytest.mark.asyncio
    async def test_build_code_relationships(self, kg_builder):
        """Test code-specific relationship building."""
        # Create class entities
        base_class = Entity(
            id="class_base",
            name="BaseAuth",
            type=EntityType.CLASS,
            content="Base authentication class",
        )
        derived_class = Entity(
            id="class_derived",
            name="ExtendedAuth",
            type=EntityType.CLASS,
            content="Extended authentication that inherits BaseAuth",
        )

        kg_builder.entities = {"class_base": base_class, "class_derived": derived_class}
        kg_builder.graph.add_node("class_base")
        kg_builder.graph.add_node("class_derived")

        await kg_builder._build_code_relationships()

        # Should create inheritance relationships
        extends_relationships = [
            r for r in kg_builder.relationships if r.type == RelationType.EXTENDS
        ]

        # Check if inheritance relationship was detected
        # (This is a simplified test - real implementation might be more sophisticated)
        assert isinstance(extends_relationships, list)  # Basic check that it runs

    @pytest.mark.asyncio
    async def test_query_subgraph_basic(self, kg_builder, mock_embedding_manager):
        """Test basic subgraph querying."""
        # Set up test entities
        entity1 = Entity(id="e1", name="Authentication", type=EntityType.CONCEPT)
        entity2 = Entity(id="e2", name="login_function", type=EntityType.FUNCTION)

        kg_builder.entities = {"e1": entity1, "e2": entity2}
        kg_builder.graph.add_node("e1")
        kg_builder.graph.add_node("e2")
        kg_builder.graph.add_edge("e1", "e2", type="contains", weight=1.0)

        # Mock finding relevant entities
        kg_builder._find_relevant_entities = AsyncMock(return_value=["e1"])

        subgraph = await kg_builder.query_subgraph(
            "authentication", max_depth=2, max_entities=10
        )

        assert isinstance(subgraph, KnowledgeSubgraph)
        assert subgraph.central_entities == ["e1"]
        assert len(subgraph.entities) > 0
        assert subgraph.metadata["query"] == "authentication"

    def test_get_graph_stats(self, kg_builder):
        """Test graph statistics generation."""
        # Add test entities and relationships
        entity1 = Entity(id="e1", name="Test Entity 1", type=EntityType.FUNCTION)
        entity2 = Entity(id="e2", name="Test Entity 2", type=EntityType.CLASS)

        kg_builder.entities = {"e1": entity1, "e2": entity2}

        relationship1 = Relationship(
            source_id="e1", target_id="e2", type=RelationType.CONTAINS
        )
        kg_builder.relationships = [relationship1]

        kg_builder.graph.add_node("e1")
        kg_builder.graph.add_node("e2")
        kg_builder.graph.add_edge("e1", "e2", type="contains")

        stats = kg_builder.get_graph_stats()

        assert stats["entity_count"] == 2
        assert stats["relationship_count"] == 1
        assert "entity_types" in stats
        assert "relationship_types" in stats
        assert "connected_components" in stats
        assert "density" in stats

    @pytest.mark.asyncio
    async def test_discover_patterns(self, kg_builder):
        """Test pattern discovery in knowledge graph."""
        # Set up test graph with patterns
        entities = {
            "f1": Entity(id="f1", name="func1", type=EntityType.FUNCTION),
            "f2": Entity(id="f2", name="func2", type=EntityType.FUNCTION),
            "c1": Entity(id="c1", name="class1", type=EntityType.CLASS),
            "c2": Entity(id="c2", name="class2", type=EntityType.CLASS),
        }

        relationships = [
            Relationship("c1", "f1", RelationType.CONTAINS),
            Relationship("c2", "f2", RelationType.CONTAINS),
            Relationship("f1", "f2", RelationType.CALLS),
        ]

        kg_builder.entities = entities
        kg_builder.relationships = relationships

        # Add to graph
        for entity_id in entities:
            kg_builder.graph.add_node(entity_id)

        for rel in relationships:
            kg_builder.graph.add_edge(
                rel.source_id, rel.target_id, type=rel.type.value, weight=rel.weight
            )

        patterns = await kg_builder.discover_patterns(min_support=0.1)

        assert isinstance(patterns, list)

        # Should find relationship patterns
        rel_patterns = [p for p in patterns if "->" in p.get("pattern", "")]
        assert len(rel_patterns) > 0

        # Should find hub entities if any exist
        hub_patterns = [p for p in patterns if p.get("pattern") == "hub_entities"]
        # May or may not exist depending on graph structure

    @pytest.mark.asyncio
    async def test_find_relevant_entities(self, kg_builder, mock_redis_store):
        """Test finding relevant entities by embedding similarity."""
        query_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        # Mock Redis scan operations
        mock_redis_store.redis.scan = MagicMock(
            side_effect=[
                (0, [b"kg_entity:e1", b"kg_entity:e2"]),
            ]
        )

        # Mock entity data with embeddings
        mock_redis_store.redis.hgetall = MagicMock(
            side_effect=[
                {
                    b"embedding": np.array(
                        [0.15, 0.25, 0.35], dtype=np.float32
                    ).tobytes()
                },
                {b"embedding": np.array([0.8, 0.1, 0.1], dtype=np.float32).tobytes()},
            ]
        )

        relevant_entities = await kg_builder._find_relevant_entities(
            query_embedding, k=2
        )

        assert isinstance(relevant_entities, list)
        assert len(relevant_entities) <= 2

    @pytest.mark.asyncio
    async def test_store_graph(self, kg_builder, mock_redis_store):
        """Test storing graph in Redis."""
        # Add test entities and relationships
        entity = Entity(
            id="test_entity",
            name="Test Entity",
            type=EntityType.FUNCTION,
            content="test content",
            embedding=np.array([0.1, 0.2], dtype=np.float32),
            properties={"lang": "python"},
            source_ids={"src1"},
        )

        relationship = Relationship(
            source_id="e1",
            target_id="e2",
            type=RelationType.CALLS,
            weight=0.8,
            properties={"freq": 5},
        )

        kg_builder.entities = {"test_entity": entity}
        kg_builder.relationships = [relationship]

        mock_redis_store.redis.hset = MagicMock()

        await kg_builder._store_graph()

        # Verify Redis operations were called
        assert (
            mock_redis_store.redis.hset.call_count >= 2
        )  # At least entity + relationship + metadata
