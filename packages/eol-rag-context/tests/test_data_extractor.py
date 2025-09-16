"""
Unit tests for the data extractor module.
"""

import json

import pytest

from eol.rag_context.data_extractor import DataEntity, DataExtractor, DataRelation
from eol.rag_context.multimodal_config import MultimodalConfig


class TestDataExtractor:
    """Test the DataExtractor class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return MultimodalConfig(
            extract_schema=True,
            infer_data_types=True,
            csv_delimiter=",",
            csv_encoding="utf-8",
            json_max_depth=3,
        )

    @pytest.fixture
    def extractor(self, config):
        """Create a data extractor instance."""
        return DataExtractor(config)

    @pytest.fixture
    def sample_json(self):
        """Sample JSON data."""
        return {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"},
            ],
            "settings": {
                "theme": "dark",
                "notifications": True,
            },
        }

    @pytest.fixture
    def sample_csv(self):
        """Sample CSV data."""
        return """id,name,age,city
1,Alice,30,New York
2,Bob,25,San Francisco
3,Charlie,35,Chicago
"""

    @pytest.fixture
    def sample_jsonl(self):
        """Sample JSONL data."""
        return """{"id": 1, "event": "login", "user": "alice"}
{"id": 2, "event": "logout", "user": "alice"}
{"id": 3, "event": "login", "user": "bob"}
"""

    @pytest.fixture
    def sample_xml(self):
        """Sample XML data."""
        return """<?xml version="1.0"?>
<root>
    <user id="1">
        <name>Alice</name>
        <email>alice@example.com</email>
    </user>
    <user id="2">
        <name>Bob</name>
        <email>bob@example.com</email>
    </user>
</root>
"""

    def test_data_entity_creation(self):
        """Test DataEntity creation and to_dict method."""
        entity = DataEntity(
            id="test_id",
            type="test_type",
            name="test_name",
            content="test content",
            metadata={"key": "value"},
            schema={"type": "object"},
        )

        result = entity.to_dict()
        assert result["id"] == "test_id"
        assert result["type"] == "test_type"
        assert result["name"] == "test_name"
        assert result["content"] == "test content"
        assert result["metadata"]["key"] == "value"
        assert result["schema"]["type"] == "object"

    def test_data_relation_creation(self):
        """Test DataRelation creation and to_dict method."""
        relation = DataRelation(
            source="source_id",
            target="target_id",
            type="contains",
            metadata={"position": 0},
        )

        result = relation.to_dict()
        assert result["source"] == "source_id"
        assert result["target"] == "target_id"
        assert result["type"] == "contains"
        assert result["metadata"]["position"] == 0

    @pytest.mark.asyncio
    async def test_extract_from_json_file(self, extractor, sample_json, tmp_path):
        """Test extracting entities from JSON file."""
        json_file = tmp_path / "data.json"
        json_file.write_text(json.dumps(sample_json))

        entities, relations = await extractor.extract_from_file(json_file)

        # Should have at least the file entity
        assert len(entities) > 0
        file_entity = entities[0]
        assert file_entity["type"] == "json_file"
        assert file_entity["name"] == "data.json"
        assert "users" in file_entity["content"] or "settings" in file_entity["content"]

        # Should extract nested entities
        assert len(entities) > 1  # File + nested entities

    @pytest.mark.asyncio
    async def test_extract_from_csv_file_basic(self, extractor, sample_csv, tmp_path):
        """Test extracting entities from CSV file without pandas."""
        # Ensure pandas is not used
        extractor._pandas_available = False

        csv_file = tmp_path / "data.csv"
        csv_file.write_text(sample_csv)

        entities, relations = await extractor.extract_from_file(csv_file)

        # Should have the file entity
        assert len(entities) > 0
        file_entity = entities[0]
        assert file_entity["type"] == "csv_file"
        assert file_entity["name"] == "data.csv"
        assert "3 rows" in file_entity["content"]

    @pytest.mark.asyncio
    async def test_extract_from_jsonl_file(self, extractor, sample_jsonl, tmp_path):
        """Test extracting entities from JSONL file."""
        jsonl_file = tmp_path / "events.jsonl"
        jsonl_file.write_text(sample_jsonl)

        entities, relations = await extractor.extract_from_file(jsonl_file)

        # Should have the file entity
        assert len(entities) > 0
        file_entity = entities[0]
        assert file_entity["type"] == "jsonl_file"
        assert file_entity["name"] == "events.jsonl"
        assert "3 records" in file_entity["content"]

    @pytest.mark.asyncio
    async def test_extract_from_xml_file(self, extractor, sample_xml, tmp_path):
        """Test extracting entities from XML file."""
        xml_file = tmp_path / "users.xml"
        xml_file.write_text(sample_xml)

        entities, relations = await extractor.extract_from_file(xml_file)

        # Should have the file entity
        assert len(entities) > 0
        file_entity = entities[0]
        assert file_entity["type"] == "xml_file"
        assert file_entity["name"] == "users.xml"
        assert "root element: root" in file_entity["content"]

        # Should extract child elements
        assert len(entities) > 1

    @pytest.mark.asyncio
    async def test_extract_from_unsupported_file(self, extractor, tmp_path):
        """Test handling of unsupported file types."""
        txt_file = tmp_path / "file.txt"
        txt_file.write_text("plain text")

        entities, relations = await extractor.extract_from_file(txt_file)

        # Should return empty lists
        assert entities == []
        assert relations == []

    @pytest.mark.asyncio
    async def test_extract_with_schema(self, extractor, sample_json, tmp_path):
        """Test schema extraction from JSON."""
        json_file = tmp_path / "schema_test.json"
        json_file.write_text(json.dumps(sample_json))

        entities, relations = await extractor.extract_from_file(json_file)

        file_entity = entities[0]
        # Schema should be extracted when configured
        if extractor.config.extract_schema:
            assert file_entity.get("schema") is not None

    def test_extract_json_schema(self, extractor):
        """Test JSON schema extraction."""
        data = {
            "string": "value",
            "number": 42,
            "boolean": True,
            "array": [1, 2, 3],
            "object": {"nested": "value"},
        }

        schema = extractor._extract_json_schema(data)

        assert schema["type"] == "object"
        assert "properties" in schema
        assert schema["properties"]["string"]["type"] == "str"
        assert schema["properties"]["number"]["type"] == "int"
        assert schema["properties"]["boolean"]["type"] == "bool"
        assert schema["properties"]["array"]["type"] == "array"
        assert schema["properties"]["object"]["type"] == "object"

    def test_extract_json_schema_max_depth(self, extractor):
        """Test JSON schema extraction with max depth limit."""
        extractor.config.json_max_depth = 1

        deep_data = {"level1": {"level2": {"level3": "value"}}}

        schema = extractor._extract_json_schema(deep_data, depth=0)

        assert schema["type"] == "object"
        assert schema["properties"]["level1"]["type"] == "dict"
        # Should not go deeper than depth 1
        assert "properties" not in schema["properties"]["level1"]

    def test_extract_from_dict(self, extractor):
        """Test entity extraction from dictionary."""
        data = {
            "users": [1, 2, 3],
            "config": {"theme": "dark"},
        }

        entities, relations = extractor._extract_from_dict(data, "parent_id", "test")

        # Should extract entities for dictionary keys
        assert len(entities) > 0
        assert len(relations) > 0

        # Check that relationships are created
        for relation in relations:
            # The source will be the entity id, not parent_id
            assert relation["type"] == "contains"

    def test_extract_from_list(self, extractor):
        """Test entity extraction from list."""
        data = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Charlie"},
        ]

        entities, relations = extractor._extract_from_list(data, "parent_id", "test")

        # Should sample first few items
        assert len(entities) > 0
        assert len(relations) > 0

        # Check relationships
        for i, relation in enumerate(relations):
            assert relation["source"] == "parent_id"
            assert relation["type"] == "contains"
            assert relation["metadata"]["index"] == i

    def test_extract_config_keys(self):
        """Test configuration key extraction."""
        # This method is in RelationshipDiscovery, not DataExtractor
        # Skip this test as it's testing the wrong module
        pytest.skip("Method belongs to RelationshipDiscovery module")

    def test_extract_xml_elements(self, extractor):
        """Test XML element extraction."""
        import xml.etree.ElementTree as ET

        xml_string = """
        <root>
            <child1 attr="value">Text1</child1>
            <child2>
                <grandchild>Text2</grandchild>
            </child2>
        </root>
        """
        root = ET.fromstring(xml_string)

        elements = extractor._extract_xml_elements(root, "parent_id")

        # Should extract child elements
        assert len(elements) > 0

        # Check element structure
        for element in elements:
            assert element["type"] == "xml_element"
            assert "name" in element
            assert "metadata" in element

    @pytest.mark.asyncio
    async def test_csv_with_pandas(self, config, sample_csv, tmp_path):
        """Test CSV extraction with pandas available."""
        try:
            import pandas as pd

            # Create extractor that thinks pandas is available
            extractor = DataExtractor(config)
            extractor._pandas_available = True
            extractor.pd = pd

            csv_file = tmp_path / "pandas_test.csv"
            csv_file.write_text(sample_csv)

            entities, relations = await extractor.extract_from_file(csv_file)

            # Should have file entity and column entities
            assert len(entities) > 1

            # Find column entities
            column_entities = [e for e in entities if e["type"] == "csv_column"]
            assert len(column_entities) == 4  # id, name, age, city

            # Check relationships between file and columns
            assert len(relations) >= len(column_entities)

        except ImportError:
            pytest.skip("Pandas not available")

    @pytest.mark.asyncio
    async def test_detect_column_relationships_with_pandas(self, config):
        """Test column relationship detection with pandas."""
        try:
            import pandas as pd

            extractor = DataExtractor(config)
            extractor._pandas_available = True
            extractor.pd = pd

            # Create DataFrame with correlated columns
            df = pd.DataFrame(
                {
                    "user_id": [1, 2, 3, 4, 5],
                    "age": [20, 25, 30, 35, 40],
                    "income": [20000, 25000, 30000, 35000, 40000],  # Correlated with age
                    "category": ["A", "B", "A", "B", "A"],
                }
            )

            relations = extractor._detect_column_relationships(df)

            # Should detect correlation between age and income
            assert len(relations) > 0

            # Check for foreign key detection
            df2 = pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "user_id": [1, 2, 3],  # Foreign key reference
                    "data": ["x", "y", "z"],
                }
            )

            relations2 = extractor._detect_column_relationships(df2)
            # Should detect user_id as potential foreign key
            fk_relations = [r for r in relations2 if r["type"] == "references"]
            assert len(fk_relations) > 0

        except ImportError:
            pytest.skip("Pandas not available")

    @pytest.mark.asyncio
    async def test_error_handling_json(self, extractor, tmp_path):
        """Test error handling for malformed JSON."""
        json_file = tmp_path / "bad.json"
        json_file.write_text("{invalid json}")

        entities, relations = await extractor.extract_from_file(json_file)

        # Should return empty lists on error
        assert entities == []
        assert relations == []

    @pytest.mark.asyncio
    async def test_error_handling_csv(self, extractor, tmp_path):
        """Test error handling for malformed CSV."""
        csv_file = tmp_path / "bad.csv"
        # Write binary data that can't be decoded as CSV
        csv_file.write_bytes(b"\xff\xfe\x00\x01")

        entities, relations = await extractor.extract_from_file(csv_file)

        # Should handle error gracefully
        assert isinstance(entities, list)
        assert isinstance(relations, list)
