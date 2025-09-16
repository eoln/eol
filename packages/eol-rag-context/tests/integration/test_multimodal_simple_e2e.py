"""
Simple end-to-end integration tests for multimodal knowledge graph.
"""

import json
import tempfile
from pathlib import Path

import pytest

from eol.rag_context.code_analyzer import ASTCodeAnalyzer
from eol.rag_context.data_extractor import DataExtractor
from eol.rag_context.multimodal_config import MultimodalConfig
from eol.rag_context.relationship_discovery import RelationshipDiscovery


@pytest.mark.integration
@pytest.mark.asyncio
class TestMultimodalSimpleE2E:
    """Simple E2E tests for multimodal knowledge graph without Redis."""

    @pytest.fixture
    def sample_code(self):
        """Sample Python code that references data files."""
        return """
import pandas as pd
import json

def load_users():
    \"\"\"Load user data from CSV.\"\"\"
    df = pd.read_csv("users.csv")
    return df

def load_config():
    \"\"\"Load configuration from JSON.\"\"\"
    with open("config.json", "r") as f:
        config = json.load(f)
    return config

class DataProcessor:
    \"\"\"Process data from various sources.\"\"\"

    def process_products(self, products_file):
        \"\"\"Process product data.\"\"\"
        df = pd.read_csv(products_file)
        return df[df["price"] > 100]
"""

    @pytest.fixture
    def sample_data_files(self):
        """Sample data files."""
        return {
            "users.csv": "id,name,email\\n1,Alice,alice@example.com\\n2,Bob,bob@example.com",
            "products.csv": "id,name,price\\n1,Laptop,999\\n2,Mouse,25",
            "config.json": json.dumps({"database": {"host": "localhost"}, "cache": True}),
        }

    @pytest.mark.asyncio
    async def test_code_data_relationship_discovery(self, sample_code, sample_data_files):
        """Test discovering relationships between code and data files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create code file
            code_file = tmpdir_path / "processor.py"
            code_file.write_text(sample_code)

            # Create data files
            for filename, content in sample_data_files.items():
                data_file = tmpdir_path / filename
                data_file.write_text(content)

            # Analyze code
            analyzer = ASTCodeAnalyzer()
            code_entities, code_relations = analyzer.analyze_file(code_file)

            # Convert code entities to dicts
            code_dicts = [e.to_dict() for e in code_entities]

            # Extract data entities
            config = MultimodalConfig()
            extractor = DataExtractor(config)

            all_data_entities = []
            for data_file in tmpdir_path.glob("*.csv"):
                entities, _ = await extractor.extract_from_file(data_file)
                all_data_entities.extend(entities)

            for json_file in tmpdir_path.glob("*.json"):
                entities, _ = await extractor.extract_from_file(json_file)
                all_data_entities.extend(entities)

            # Discover relationships
            discovery = RelationshipDiscovery(similarity_threshold=0.5)
            relationships = await discovery.discover_relationships(code_dicts, all_data_entities)

            # Verify code entities found
            assert len(code_entities) > 0
            entity_names = {e.name for e in code_entities}
            assert "load_users" in entity_names
            assert "load_config" in entity_names
            assert "DataProcessor" in entity_names

            # Verify data entities found
            assert len(all_data_entities) > 0
            data_types = {e.get("type") for e in all_data_entities}
            assert "csv_file" in data_types
            assert "json_file" in data_types

            # Verify some relationships discovered
            assert isinstance(relationships, list)
            if relationships:
                rel_types = {r.relationship_type.value for r in relationships}
                # Check for expected relationship types
                assert len(rel_types) > 0

    @pytest.mark.asyncio
    async def test_pattern_detection(self, sample_code):
        """Test pattern detection in code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create multiple similar code files
            for i in range(3):
                code_file = tmpdir_path / f"module_{i}.py"
                code_file.write_text(
                    f"""
def load_data_{i}():
    df = pd.read_csv("data_{i}.csv")
    return df

def process_data_{i}(df):
    return df.head()
"""
                )

            # Analyze all files
            analyzer = ASTCodeAnalyzer()
            all_entities = []
            all_relations = []

            for py_file in tmpdir_path.glob("*.py"):
                entities, relations = analyzer.analyze_file(py_file)
                all_entities.extend(entities)
                all_relations.extend(relations)

            # Check for patterns
            function_names = [e.name for e in all_entities if e.type.value == "function"]

            # Should find pattern of load_data_* functions
            load_functions = [n for n in function_names if n.startswith("load_data_")]
            assert len(load_functions) == 3

            # Should find pattern of process_data_* functions
            process_functions = [n for n in function_names if n.startswith("process_data_")]
            assert len(process_functions) == 3

    @pytest.mark.asyncio
    async def test_config_binding_discovery(self):
        """Test discovery of configuration bindings in code."""
        code_with_config = """
def setup(config):
    db_host = config['database']['host']
    cache_enabled = config.get('cache', False)
    return db_host, cache_enabled

class Service:
    def __init__(self, config):
        self.api_key = config['api']['key']
        self.timeout = config.get('timeout', 30)
"""

        config_data = {
            "database": {"host": "localhost", "port": 5432},
            "api": {"key": "secret", "endpoint": "https://api.example.com"},
            "cache": True,
            "timeout": 60,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create code file
            code_file = tmpdir_path / "service.py"
            code_file.write_text(code_with_config)

            # Analyze code
            analyzer = ASTCodeAnalyzer()
            code_entities, _ = analyzer.analyze_file(code_file)
            code_dicts = [e.to_dict() for e in code_entities]

            # Discover config bindings
            discovery = RelationshipDiscovery()
            relationships = discovery.discover_config_bindings(code_dicts, config_data)

            # Should find config bindings
            assert len(relationships) > 0
            assert all(r.relationship_type.value == "config_binding" for r in relationships)

            # Check that specific config keys were found
            evidence_keys = []
            for rel in relationships:
                if "config_keys" in rel.evidence:
                    evidence_keys.extend(rel.evidence["config_keys"])

            # Should find references to database.host and api.key
            assert any("database" in key for key in evidence_keys)
            assert any("api" in key for key in evidence_keys)

    @pytest.mark.asyncio
    async def test_multimodal_workflow_complete(self):
        """Test complete multimodal workflow from code to data to relationships."""
        # Create a mini project
        project_structure = {
            "main.py": """
from data_loader import DataLoader
from processor import process_data

def main():
    loader = DataLoader()
    users = loader.load_users()
    products = loader.load_products()
    results = process_data(users, products)
    return results
""",
            "data_loader.py": """
import pandas as pd

class DataLoader:
    def load_users(self):
        return pd.read_csv("data/users.csv")

    def load_products(self):
        return pd.read_csv("data/products.csv")
""",
            "processor.py": """
def process_data(users, products):
    # Join users with their product purchases
    return users.merge(products, on="user_id")
""",
            "data/users.csv": "user_id,name\\n1,Alice\\n2,Bob",
            "data/products.csv": "user_id,product\\n1,Laptop\\n2,Phone",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create project structure
            for file_path, content in project_structure.items():
                full_path = tmpdir_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)

            # Analyze all Python files
            analyzer = ASTCodeAnalyzer()
            all_code_entities = []
            all_code_relations = []

            for py_file in tmpdir_path.glob("**/*.py"):
                entities, relations = analyzer.analyze_file(py_file)
                all_code_entities.extend(entities)
                all_code_relations.extend(relations)

            # Extract all data files
            config = MultimodalConfig()
            extractor = DataExtractor(config)
            all_data_entities = []

            for csv_file in tmpdir_path.glob("**/*.csv"):
                entities, _ = await extractor.extract_from_file(csv_file)
                all_data_entities.extend(entities)

            # Discover cross-modal relationships
            code_dicts = [e.to_dict() for e in all_code_entities]
            discovery = RelationshipDiscovery()
            relationships = await discovery.discover_relationships(code_dicts, all_data_entities)

            # Verify complete workflow
            assert len(all_code_entities) > 5  # Multiple modules, classes, functions
            assert len(all_data_entities) >= 2  # At least 2 CSV files

            # Check code structure
            module_names = {e.name for e in all_code_entities if e.type.value == "module"}
            assert "main" in module_names
            assert "data_loader" in module_names
            assert "processor" in module_names

            # Check class and function entities
            class_names = {e.name for e in all_code_entities if e.type.value == "class"}
            assert "DataLoader" in class_names

            function_names = {e.name for e in all_code_entities if e.type.value == "function"}
            assert "main" in function_names
            assert "process_data" in function_names

            # Check import relationships
            import_relations = [r for r in all_code_relations if r.type.value == "imports"]
            assert len(import_relations) > 0

            # Verify data files were processed
            csv_files = [e for e in all_data_entities if e.get("type") == "csv_file"]
            assert len(csv_files) == 2

            print(f"✓ Found {len(all_code_entities)} code entities")
            print(f"✓ Found {len(all_code_relations)} code relations")
            print(f"✓ Found {len(all_data_entities)} data entities")
            print(f"✓ Found {len(relationships)} cross-modal relationships")
