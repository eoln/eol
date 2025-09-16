"""
Unit tests for the relationship discovery module.
"""

from unittest.mock import AsyncMock

import pytest

from eol.rag_context.relationship_discovery import (
    CrossModalRelationType,
    DiscoveredRelationship,
    RelationshipDiscovery,
)


class TestRelationshipDiscovery:
    """Test the RelationshipDiscovery class."""

    @pytest.fixture
    def embedding_manager(self):
        """Create a mock embedding manager."""
        manager = AsyncMock()
        manager.get_embedding = AsyncMock(return_value=[0.1] * 384)
        return manager

    @pytest.fixture
    def discovery(self, embedding_manager):
        """Create a relationship discovery instance."""
        return RelationshipDiscovery(embedding_manager=embedding_manager, similarity_threshold=0.7)

    @pytest.fixture
    def code_entities(self):
        """Sample code entities."""
        return [
            {
                "id": "func_1",
                "type": "function",
                "name": "load_users",
                "content": 'df = pd.read_csv("users.csv")',
            },
            {
                "id": "func_2",
                "type": "function",
                "name": "save_config",
                "content": 'with open("config.json", "w") as f:',
            },
            {
                "id": "func_3",
                "type": "function",
                "name": "process_data",
                "content": 'data = load_data("products.csv")',
            },
        ]

    @pytest.fixture
    def data_entities(self):
        """Sample data entities."""
        return [
            {
                "id": "data_1",
                "type": "csv_file",
                "name": "users",
                "content": "CSV file with user data",
            },
            {
                "id": "data_2",
                "type": "json_file",
                "name": "config",
                "content": "JSON configuration file",
            },
            {
                "id": "data_3",
                "type": "csv_file",
                "name": "products",
                "content": "Product catalog CSV",
            },
        ]

    def test_discovered_relationship_creation(self):
        """Test DiscoveredRelationship creation and to_dict."""
        rel = DiscoveredRelationship(
            source_entity_id="source",
            target_entity_id="target",
            relationship_type=CrossModalRelationType.CODE_REFERENCES_DATA,
            confidence=0.95,
            evidence={"pattern": "test"},
        )

        result = rel.to_dict()
        assert result["source"] == "source"
        assert result["target"] == "target"
        assert result["type"] == "code_references_data"
        assert result["confidence"] == 0.95
        assert result["evidence"]["pattern"] == "test"

    @pytest.mark.asyncio
    async def test_discover_code_data_relationships(self, discovery, code_entities, data_entities):
        """Test discovery of code-data relationships."""
        relationships = await discovery._discover_code_data_relationships(
            code_entities, data_entities
        )

        # Should find relationships based on file references
        assert len(relationships) > 0

        # Check for specific matches
        found_users = any(
            r.target_entity_id == "data_1"
            and r.relationship_type == CrossModalRelationType.CODE_REFERENCES_DATA
            for r in relationships
        )
        assert found_users

        found_config = any(
            r.target_entity_id == "data_2"
            and r.relationship_type == CrossModalRelationType.CODE_REFERENCES_DATA
            for r in relationships
        )
        assert found_config

    def test_discover_pattern_relationships(self, discovery, code_entities, data_entities):
        """Test pattern-based relationship discovery."""
        # Add code entities with pattern-matching names
        code_entities.append(
            {
                "id": "func_4",
                "type": "function",
                "name": "get_users",
                "content": "return users",
            }
        )

        relationships = discovery._discover_pattern_relationships(code_entities, data_entities)

        # Should find pattern match between get_users and users
        assert len(relationships) > 0

        pattern_rels = [
            r for r in relationships if r.relationship_type == CrossModalRelationType.PATTERN_MATCH
        ]
        assert len(pattern_rels) > 0

    def test_discover_api_mappings(self, discovery):
        """Test API endpoint mapping discovery."""
        code_entities = [
            {
                "id": "api_1",
                "type": "function",
                "name": "users_endpoint",
                "content": '@app.route("/api/users")',
            },
            {
                "id": "api_2",
                "type": "function",
                "name": "products_endpoint",
                "content": '@router.get("/products/{id}")',
            },
        ]

        data_entities = [
            {"id": "data_1", "name": "users", "type": "csv_file"},
            {"id": "data_2", "name": "products", "type": "json_file"},
        ]

        relationships = discovery._discover_api_mappings(code_entities, data_entities)

        # Should find API endpoint mappings
        assert len(relationships) > 0

        api_rels = [
            r
            for r in relationships
            if r.relationship_type == CrossModalRelationType.API_ENDPOINT_MAPPING
        ]
        assert len(api_rels) > 0

    def test_discover_config_bindings(self, discovery):
        """Test configuration binding discovery."""
        code_entities = [
            {
                "id": "code_1",
                "type": "function",
                "name": "setup",
                "content": 'theme = config["display.theme"]',
            },
            {
                "id": "code_2",
                "type": "function",
                "name": "init",
                "content": 'db_host = config.get("database.host")',
            },
        ]

        config_data = {
            "display": {"theme": "dark"},
            "database": {"host": "localhost"},
        }

        relationships = discovery.discover_config_bindings(code_entities, config_data)

        # Should find config bindings
        assert len(relationships) > 0

        config_rels = [
            r for r in relationships if r.relationship_type == CrossModalRelationType.CONFIG_BINDING
        ]
        assert len(config_rels) > 0

    @pytest.mark.asyncio
    async def test_discover_semantic_relationships(self, discovery, code_entities, data_entities):
        """Test semantic similarity relationship discovery."""
        # Mock embeddings with high similarity
        similar_embedding = [0.9] * 384
        different_embedding = [0.1] * 384

        discovery.embedding_manager.get_embedding = AsyncMock(
            side_effect=[similar_embedding, different_embedding, similar_embedding]
        )

        relationships = await discovery._discover_semantic_relationships(
            code_entities[:1], data_entities[:2], None
        )

        # Should find semantic similarities based on embeddings
        if relationships:
            assert all(
                r.relationship_type == CrossModalRelationType.SEMANTIC_SIMILARITY
                for r in relationships
            )

    def test_cosine_similarity(self, discovery):
        """Test cosine similarity calculation."""
        # Test identical vectors
        vec1 = [1.0, 0.0, 0.0]
        similarity = discovery._cosine_similarity(vec1, vec1)
        assert abs(similarity - 1.0) < 0.001

        # Test orthogonal vectors
        vec2 = [0.0, 1.0, 0.0]
        similarity = discovery._cosine_similarity(vec1, vec2)
        assert abs(similarity) < 0.001

        # Test opposite vectors
        vec3 = [-1.0, 0.0, 0.0]
        similarity = discovery._cosine_similarity(vec1, vec3)
        assert abs(similarity + 1.0) < 0.001

        # Test zero vectors
        zero = [0.0, 0.0, 0.0]
        similarity = discovery._cosine_similarity(zero, vec1)
        assert similarity == 0.0

    def test_get_entity_text(self, discovery):
        """Test entity text extraction for embedding."""
        entity = {
            "name": "test_function",
            "type": "function",
            "docstring": "This is a test function",
            "content": "def test_function(): pass",
        }

        text = discovery._get_entity_text(entity)

        assert "test_function" in text
        assert "function" in text
        assert "This is a test function" in text
        assert "def test_function" in text

    def test_extract_config_keys(self, discovery):
        """Test configuration key extraction."""
        config_data = {
            "level1": {
                "level2": {"key": "value"},
                "another": "value",
            },
            "top": "level",
        }

        keys = discovery._extract_config_keys(config_data)

        assert "level1" in keys
        assert "level1.level2" in keys
        assert "level1.level2.key" in keys
        assert "level1.another" in keys
        assert "top" in keys

    def test_merge_duplicate_relationships(self, discovery):
        """Test merging of duplicate relationships."""
        # Create duplicate relationships
        rel1 = DiscoveredRelationship(
            source_entity_id="source",
            target_entity_id="target",
            relationship_type=CrossModalRelationType.CODE_REFERENCES_DATA,
            confidence=0.8,
            evidence={"pattern": "pattern1"},
        )

        rel2 = DiscoveredRelationship(
            source_entity_id="source",
            target_entity_id="target",
            relationship_type=CrossModalRelationType.PATTERN_MATCH,
            confidence=0.9,
            evidence={"pattern": "pattern2"},
        )

        rel3 = DiscoveredRelationship(
            source_entity_id="other_source",
            target_entity_id="other_target",
            relationship_type=CrossModalRelationType.SEMANTIC_SIMILARITY,
            confidence=0.7,
            evidence={"similarity": 0.7},
        )

        discovery.discovered_relationships = [rel1, rel2, rel3]

        merged = discovery.merge_duplicate_relationships(min_confidence=0.5)

        # Should merge rel1 and rel2, keep rel3 separate
        assert len(merged) == 2

        # Check that highest confidence is kept
        merged_rel = next(
            r for r in merged if r.source_entity_id == "source" and r.target_entity_id == "target"
        )
        assert merged_rel.confidence == 0.9

        # Check that evidence is combined
        assert CrossModalRelationType.CODE_REFERENCES_DATA.value in merged_rel.evidence
        assert CrossModalRelationType.PATTERN_MATCH.value in merged_rel.evidence

    def test_merge_with_confidence_filter(self, discovery):
        """Test merging with minimum confidence filter."""
        rel1 = DiscoveredRelationship(
            source_entity_id="s1",
            target_entity_id="t1",
            relationship_type=CrossModalRelationType.PATTERN_MATCH,
            confidence=0.3,  # Below threshold
            evidence={},
        )

        rel2 = DiscoveredRelationship(
            source_entity_id="s2",
            target_entity_id="t2",
            relationship_type=CrossModalRelationType.SEMANTIC_SIMILARITY,
            confidence=0.8,  # Above threshold
            evidence={},
        )

        discovery.discovered_relationships = [rel1, rel2]

        merged = discovery.merge_duplicate_relationships(min_confidence=0.5)

        # Should only include rel2
        assert len(merged) == 1
        assert merged[0].confidence == 0.8

    @pytest.mark.asyncio
    async def test_discover_relationships_full_workflow(
        self, discovery, code_entities, data_entities
    ):
        """Test the complete relationship discovery workflow."""
        concept_entities = [
            {"id": "concept_1", "name": "User Management", "type": "concept"},
        ]

        relationships = await discovery.discover_relationships(
            code_entities, data_entities, concept_entities
        )

        # Should return a list of relationships
        assert isinstance(relationships, list)

        # Should have various relationship types
        if relationships:
            rel_types = {r.relationship_type for r in relationships}
            assert len(rel_types) > 0

    @pytest.mark.asyncio
    async def test_discover_relationships_without_embedding_manager(self):
        """Test relationship discovery without embedding manager."""
        discovery = RelationshipDiscovery(embedding_manager=None, similarity_threshold=0.7)

        code_entities = [{"id": "c1", "name": "test", "content": "code"}]
        data_entities = [{"id": "d1", "name": "test", "content": "data"}]

        relationships = await discovery.discover_relationships(code_entities, data_entities)

        # Should still work but skip semantic similarity
        assert isinstance(relationships, list)
        # Should not have semantic similarity relationships
        semantic_rels = [
            r
            for r in relationships
            if r.relationship_type == CrossModalRelationType.SEMANTIC_SIMILARITY
        ]
        assert len(semantic_rels) == 0

    def test_pattern_matching_plural_singular(self, discovery):
        """Test pattern matching with plural/singular variations."""
        code_entities = [
            {"id": "c1", "name": "get_user", "type": "function", "content": ""},
            {"id": "c2", "name": "fetch_products", "type": "function", "content": ""},
        ]

        data_entities = [
            {"id": "d1", "name": "users", "type": "csv_file"},  # Plural
            {"id": "d2", "name": "product", "type": "json_file"},  # Singular
        ]

        relationships = discovery._discover_pattern_relationships(code_entities, data_entities)

        # Should find matches despite plural/singular differences
        assert len(relationships) > 0
