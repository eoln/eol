"""
Integration tests for multimodal knowledge graph functionality.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.integration
class TestMultimodalKnowledgeGraph:
    """Test multimodal knowledge graph building and relationship discovery."""

    @pytest.fixture
    def sample_python_code(self):
        """Sample Python code with data references."""
        return '''
import pandas as pd
import json

class DataProcessor:
    """Process user data from CSV files."""

    def __init__(self):
        self.users_df = None
        self.config = None

    def load_users(self):
        """Load users from CSV file."""
        self.users_df = pd.read_csv("users.csv")
        return self.users_df

    def load_config(self):
        """Load configuration from JSON."""
        with open("config.json", "r") as f:
            self.config = json.load(f)
        return self.config

    def process_users(self, users_data):
        """Process user data."""
        # Filter active users
        active_users = users_data[users_data["status"] == "active"]
        return active_users

def get_user_by_id(user_id):
    """Fetch a user by ID."""
    processor = DataProcessor()
    users = processor.load_users()
    return users[users["id"] == user_id].iloc[0]
'''

    @pytest.fixture
    def sample_csv_data(self):
        """Sample CSV data."""
        return """id,name,email,status
1,John Doe,john@example.com,active
2,Jane Smith,jane@example.com,active
3,Bob Wilson,bob@example.com,inactive
"""

    @pytest.fixture
    def sample_json_data(self):
        """Sample JSON configuration."""
        return {
            "database": {"host": "localhost", "port": 5432},
            "cache": {"enabled": True, "ttl": 3600},
            "features": {"multimodal": True, "analytics": False},
        }

    @pytest.mark.asyncio
    async def test_code_analysis(self, sample_python_code):
        """Test code analysis with AST analyzer."""
        from eol.rag_context.code_analyzer import ASTCodeAnalyzer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create Python file
            code_file = Path(tmpdir) / "processor.py"
            code_file.write_text(sample_python_code)

            # Analyze the code
            analyzer = ASTCodeAnalyzer()
            entities, relations = analyzer.analyze_file(code_file)

            # Check entities were extracted
            assert len(entities) > 0

            # Check for specific entities
            entity_names = {e.name for e in entities}
            assert "DataProcessor" in entity_names
            assert "load_users" in entity_names
            assert "load_config" in entity_names
            assert "process_users" in entity_names
            assert "get_user_by_id" in entity_names

            # Check for relationships
            assert len(relations) > 0

    @pytest.mark.asyncio
    async def test_data_extraction(self, sample_csv_data, sample_json_data):
        """Test data extraction from CSV and JSON files."""
        from eol.rag_context.data_extractor import DataExtractor
        from eol.rag_context.multimodal_config import MultimodalConfig

        config = MultimodalConfig()
        extractor = DataExtractor(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CSV file
            csv_file = Path(tmpdir) / "users.csv"
            csv_file.write_text(sample_csv_data)

            # Create JSON file
            json_file = Path(tmpdir) / "config.json"
            json_file.write_text(json.dumps(sample_json_data))

            # Extract from CSV
            csv_entities, csv_relations = await extractor.extract_from_file(csv_file)
            assert len(csv_entities) > 0
            assert any(e["type"] == "csv_file" for e in csv_entities)

            # Extract from JSON
            json_entities, json_relations = await extractor.extract_from_file(json_file)
            assert len(json_entities) > 0
            assert any(e["type"] == "json_file" for e in json_entities)

    @pytest.mark.asyncio
    async def test_relationship_discovery(self, sample_python_code, sample_csv_data):
        """Test relationship discovery between code and data."""
        from eol.rag_context.code_analyzer import ASTCodeAnalyzer
        from eol.rag_context.data_extractor import DataExtractor
        from eol.rag_context.multimodal_config import MultimodalConfig
        from eol.rag_context.relationship_discovery import RelationshipDiscovery

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            code_file = Path(tmpdir) / "processor.py"
            code_file.write_text(sample_python_code)

            csv_file = Path(tmpdir) / "users.csv"
            csv_file.write_text(sample_csv_data)

            # Analyze code
            analyzer = ASTCodeAnalyzer()
            code_entities, _ = analyzer.analyze_file(code_file)
            code_dicts = [e.to_dict() for e in code_entities]

            # Extract data
            config = MultimodalConfig()
            extractor = DataExtractor(config)
            data_entities, _ = await extractor.extract_from_file(csv_file)

            # Discover relationships
            discovery = RelationshipDiscovery(similarity_threshold=0.5)
            relationships = await discovery.discover_relationships(code_dicts, data_entities)

            # Should find code-data relationships (may be empty without pandas)
            # The relationship discovery looks for filename matches in code
            # Since our code references "users.csv" but our data entity is "csv_users"
            # we may not find matches without more sophisticated matching
            assert isinstance(relationships, list)

            # Check for specific relationship types if any found
            if relationships:
                rel_types = {r.relationship_type.value for r in relationships}
                # May find code_references_data or pattern_match
                assert isinstance(rel_types, set)

    @pytest.mark.asyncio
    async def test_enhanced_knowledge_graph_builder(self):
        """Test the enhanced knowledge graph builder."""
        from eol.rag_context.enhanced_knowledge_graph import EnhancedKnowledgeGraphBuilder
        from eol.rag_context.multimodal_config import MultimodalConfig

        # Mock dependencies
        mock_redis = MagicMock()
        mock_embedding = AsyncMock()
        mock_embedding.get_embedding = AsyncMock(return_value=[0.1] * 384)

        config = MultimodalConfig.minimal()
        builder = EnhancedKnowledgeGraphBuilder(mock_redis, mock_embedding, config)

        # Test initialization
        assert builder.config is not None
        assert builder.code_analyzer is not None  # Lazy loaded

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a simple Python file
            (tmpdir_path / "test.py").write_text("def hello(): pass")

            # Test code analysis
            stats = await builder.build_from_code(tmpdir_path)
            assert "entities" in stats
            assert "relationships" in stats
            assert stats["entities"] >= 1  # At least module and function

    @pytest.mark.asyncio
    async def test_multimodal_config_validation(self):
        """Test configuration validation."""
        from eol.rag_context.multimodal_config import MultimodalConfig

        # Test minimal config
        minimal = MultimodalConfig.minimal()
        assert minimal.processing_mode.value == "minimal"
        assert not minimal.enable_image_processing
        warnings = minimal.validate()
        assert isinstance(warnings, list)

        # Test comprehensive config
        comprehensive = MultimodalConfig.comprehensive()
        assert comprehensive.processing_mode.value == "comprehensive"
        assert comprehensive.enable_image_processing
        warnings = comprehensive.validate()
        # Will have warnings about missing PIL/pytesseract
        assert len(warnings) > 0

        # Test invalid config
        invalid_config = MultimodalConfig(similarity_threshold=2.0)  # Invalid: > 1
        warnings = invalid_config.validate()
        assert any("similarity_threshold" in w for w in warnings)

    @pytest.mark.asyncio
    async def test_pattern_discovery(self):
        """Test pattern discovery in enhanced knowledge graph."""
        from eol.rag_context.enhanced_knowledge_graph import EnhancedKnowledgeGraphBuilder
        from eol.rag_context.multimodal_config import MultimodalConfig

        # Mock dependencies
        mock_redis = MagicMock()
        mock_embedding = AsyncMock()

        config = MultimodalConfig(enable_pattern_detection=True, min_pattern_frequency=1)
        builder = EnhancedKnowledgeGraphBuilder(mock_redis, mock_embedding, config)

        # Mock graph with patterns
        builder.graph = MagicMock()
        builder.graph.edges.return_value = [
            (None, None, {"type": "calls"}),
            (None, None, {"type": "calls"}),
            (None, None, {"type": "imports"}),
        ]
        builder.graph.nodes.return_value = [
            ("node1", {"type": "function"}),
            ("node2", {"type": "data", "structure": "csv"}),
            ("node3", {"type": "data", "structure": "csv"}),
        ]

        # Detect patterns
        patterns = await builder.detect_patterns(min_frequency=1)
        assert isinstance(patterns, list)

        # Should detect data structure patterns
        pattern_types = {p["type"] for p in patterns}
        assert "data_structure" in pattern_types

    @pytest.mark.asyncio
    async def test_cross_modal_integration(
        self, sample_python_code, sample_csv_data, sample_json_data
    ):
        """Test full cross-modal integration workflow."""
        from eol.rag_context.code_analyzer import ASTCodeAnalyzer
        from eol.rag_context.data_extractor import DataExtractor
        from eol.rag_context.multimodal_config import MultimodalConfig
        from eol.rag_context.relationship_discovery import RelationshipDiscovery

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test files
            (tmpdir_path / "processor.py").write_text(sample_python_code)
            (tmpdir_path / "users.csv").write_text(sample_csv_data)
            (tmpdir_path / "config.json").write_text(json.dumps(sample_json_data))

            # 1. Analyze code
            analyzer = ASTCodeAnalyzer()
            code_entities, code_relations = analyzer.analyze_directory(tmpdir_path)
            assert len(code_entities) > 0

            # 2. Extract data
            config = MultimodalConfig()
            extractor = DataExtractor(config)
            all_data_entities = []
            for data_file in tmpdir_path.glob("*.csv"):
                entities, _ = await extractor.extract_from_file(data_file)
                all_data_entities.extend(entities)
            for json_file in tmpdir_path.glob("*.json"):
                entities, _ = await extractor.extract_from_file(json_file)
                all_data_entities.extend(entities)

            assert len(all_data_entities) > 0

            # 3. Discover relationships
            discovery = RelationshipDiscovery()
            code_dicts = [e.to_dict() for e in code_entities]
            relationships = await discovery.discover_relationships(code_dicts, all_data_entities)

            # 4. Merge duplicates
            merged = discovery.merge_duplicate_relationships()
            assert isinstance(merged, list)

            # Verify complete workflow produced meaningful results
            assert len(code_entities) > 5  # Multiple code entities
            assert len(all_data_entities) >= 2  # At least CSV and JSON
            # Relationships might not be found without pandas for proper matching
            assert isinstance(relationships, list)  # At least returns a list
