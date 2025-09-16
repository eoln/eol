"""
End-to-end integration tests for multimodal knowledge graph with Redis.
"""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest
from redis.asyncio import Redis

from eol.rag_context.code_analyzer import ASTCodeAnalyzer
from eol.rag_context.config import EmbeddingConfig, IndexConfig, RedisConfig
from eol.rag_context.data_extractor import DataExtractor
from eol.rag_context.embeddings import EmbeddingManager
from eol.rag_context.enhanced_knowledge_graph import EnhancedKnowledgeGraphBuilder
from eol.rag_context.indexer import DocumentIndexer
from eol.rag_context.knowledge_graph import KnowledgeGraphBuilder
from eol.rag_context.multimodal_config import MultimodalConfig
from eol.rag_context.redis_client import RedisVectorStore
from eol.rag_context.relationship_discovery import RelationshipDiscovery


@pytest.mark.integration
@pytest.mark.asyncio
class TestMultimodalE2E:
    """End-to-end tests for multimodal knowledge graph functionality."""

    @pytest.fixture
    async def redis_store(self):
        """Create a Redis vector store for testing."""
        # Create configurations
        redis_config = RedisConfig(host="localhost", port=6379, db=15)

        index_config = IndexConfig(
            vectorset_name="test_context", prefix="test:", m=16, ef_construction=200
        )

        # Create the store
        store = RedisVectorStore(redis_config, index_config)

        # Connect to Redis
        await store.connect_async()

        # Clean up test database
        if store.async_redis:
            await store.async_redis.flushdb()

        yield store

        # Cleanup after test
        if store.async_redis:
            await store.async_redis.flushdb()
            await store.async_redis.close()

    @pytest.fixture
    async def embedding_manager(self):
        """Create an embedding manager for testing."""
        embedding_config = EmbeddingConfig(
            provider="sentence-transformers", model_name="all-MiniLM-L6-v2", dimension=384
        )
        return EmbeddingManager(embedding_config)

    @pytest.fixture
    def sample_codebase(self):
        """Create sample code files for testing."""
        code = {
            "main.py": """
import pandas as pd
from data_processor import DataProcessor
from api import create_app

def main():
    \"\"\"Main application entry point.\"\"\"
    # Load configuration
    config = load_config("config.json")

    # Initialize data processor
    processor = DataProcessor(config)

    # Load and process data
    users_df = pd.read_csv("data/users.csv")
    processed_data = processor.process_users(users_df)

    # Start API server
    app = create_app(processed_data, config)
    app.run()

def load_config(config_path):
    \"\"\"Load configuration from JSON file.\"\"\"
    with open(config_path, 'r') as f:
        return json.load(f)
""",
            "data_processor.py": """
class DataProcessor:
    \"\"\"Process user data for the application.\"\"\"

    def __init__(self, config):
        self.config = config
        self.cache = {}

    def process_users(self, users_df):
        \"\"\"Process user DataFrame.\"\"\"
        # Filter active users
        active_users = users_df[users_df['status'] == 'active']

        # Apply transformations
        if self.config.get('normalize_names'):
            active_users['name'] = active_users['name'].str.title()

        return active_users

    def get_user_stats(self, users_df):
        \"\"\"Calculate user statistics.\"\"\"
        return {
            'total': len(users_df),
            'active': len(users_df[users_df['status'] == 'active']),
            'inactive': len(users_df[users_df['status'] == 'inactive'])
        }
""",
            "api.py": """
from flask import Flask, jsonify

def create_app(data, config):
    \"\"\"Create Flask API application.\"\"\"
    app = Flask(__name__)

    @app.route('/api/users')
    def get_users():
        \"\"\"Get all users.\"\"\"
        return jsonify(data.to_dict('records'))

    @app.route('/api/stats')
    def get_stats():
        \"\"\"Get user statistics.\"\"\"
        return jsonify({
            'total_users': len(data),
            'config': config
        })

    return app
""",
        }

        data_files = {
            "data/users.csv": """id,name,email,status,created_at
1,Alice Johnson,alice@example.com,active,2024-01-01
2,Bob Smith,bob@example.com,active,2024-01-02
3,Charlie Brown,charlie@example.com,inactive,2024-01-03
4,Diana Prince,diana@example.com,active,2024-01-04
5,Eve Wilson,eve@example.com,active,2024-01-05
""",
            "data/products.csv": """id,name,price,category
101,Laptop,999.99,Electronics
102,Desk Chair,199.99,Furniture
103,Monitor,299.99,Electronics
104,Keyboard,79.99,Electronics
""",
            "config.json": json.dumps(
                {
                    "app_name": "User Management System",
                    "version": "1.0.0",
                    "database": {"host": "localhost", "port": 5432, "name": "users_db"},
                    "api": {"port": 8080, "debug": True},
                    "features": {"normalize_names": True, "cache_enabled": True},
                },
                indent=2,
            ),
        }

        return code, data_files

    @pytest.mark.asyncio
    async def test_full_multimodal_pipeline(self, redis_store, embedding_manager, sample_codebase):
        """Test the complete multimodal knowledge graph pipeline with Redis."""
        code_files, data_files = sample_codebase

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create directory structure
            (tmpdir_path / "data").mkdir(exist_ok=True)

            # Write code files
            for filename, content in code_files.items():
                file_path = tmpdir_path / filename
                file_path.write_text(content)

            # Write data files
            for filename, content in data_files.items():
                file_path = tmpdir_path / filename
                file_path.parent.mkdir(exist_ok=True)
                file_path.write_text(content)

            # 1. Build enhanced knowledge graph
            config = MultimodalConfig(
                enable_code_analysis=True,
                enable_data_extraction=True,
                enable_pattern_detection=True,
                similarity_threshold=0.7,
            )

            builder = EnhancedKnowledgeGraphBuilder(
                redis_store.async_redis, embedding_manager, config
            )

            # Analyze code
            code_stats = await builder.build_from_code(tmpdir_path)
            assert code_stats["entities"] > 0
            assert code_stats["relationships"] > 0

            # Extract data entities
            for data_file in tmpdir_path.glob("data/*.csv"):
                data_stats = await builder.build_from_data(data_file)
                assert data_stats["entities"] > 0

            config_file = tmpdir_path / "config.json"
            config_stats = await builder.build_from_data(config_file)
            assert config_stats["entities"] > 0

            # Detect patterns
            patterns = await builder.detect_patterns(min_frequency=1)
            assert len(patterns) > 0

            # 2. Index documents for RAG
            indexer = DocumentIndexer(redis_store, embedding_manager)

            # Index code files
            for py_file in tmpdir_path.glob("*.py"):
                await indexer.index_file(str(py_file), source_name="codebase")

            # Index data files
            for csv_file in tmpdir_path.glob("data/*.csv"):
                await indexer.index_file(str(csv_file), source_name="data")

            # 3. Test cross-modal search
            # Search for code that processes users
            results = await redis_store.search_similar(
                "process user data", k=5, source_filter="codebase"
            )
            assert len(results) > 0
            assert any("process_users" in r.get("content", "") for r in results)

            # Search for data files
            results = await redis_store.search_similar(
                "user information csv", k=3, source_filter="data"
            )
            assert len(results) > 0

            # 4. Test knowledge graph queries
            # Get all entities from the graph
            graph_nodes = list(builder.graph.nodes())
            assert len(graph_nodes) > 0

            # Check for specific entity types
            code_entities = [
                n for n in graph_nodes if builder.graph.nodes[n].get("type") == "function"
            ]
            data_entities = [
                n for n in graph_nodes if builder.graph.nodes[n].get("type") == "data_file"
            ]

            assert len(code_entities) > 0
            assert len(data_entities) > 0

            # 5. Test relationship discovery
            relationships = list(builder.graph.edges(data=True))
            assert len(relationships) > 0

            # Check for code-data relationships
            code_data_rels = [
                r
                for r in relationships
                if r[2].get("type") in ["references", "code_references_data"]
            ]
            assert len(code_data_rels) > 0

    @pytest.mark.asyncio
    async def test_incremental_knowledge_graph_update(
        self, redis_store, embedding_manager, sample_codebase
    ):
        """Test incremental updates to the knowledge graph."""
        code_files, data_files = sample_codebase

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Initial setup with just main.py
            main_file = tmpdir_path / "main.py"
            main_file.write_text(code_files["main.py"])

            config = MultimodalConfig.minimal()
            builder = EnhancedKnowledgeGraphBuilder(
                redis_store.async_redis, embedding_manager, config
            )

            # Build initial graph
            stats1 = await builder.build_from_code(tmpdir_path)
            initial_entities = stats1["entities"]

            # Add more files incrementally
            (tmpdir_path / "data_processor.py").write_text(code_files["data_processor.py"])

            # Update graph
            stats2 = await builder.build_from_code(tmpdir_path)
            assert stats2["entities"] > initial_entities

            # Add API file
            (tmpdir_path / "api.py").write_text(code_files["api.py"])

            # Final update
            stats3 = await builder.build_from_code(tmpdir_path)
            assert stats3["entities"] > stats2["entities"]
            assert stats3["relationships"] > stats2["relationships"]

    @pytest.mark.asyncio
    async def test_multimodal_search_with_filters(self, redis_store, embedding_manager):
        """Test searching with multimodal filters."""
        # Create and index test documents
        indexer = DocumentIndexer(redis_store, embedding_manager)

        # Index code document
        await indexer.index_document(
            content="def process_users(df): return df[df['active'] == True]",
            metadata={"type": "code", "language": "python", "file": "processor.py"},
            doc_id="code_1",
            source_name="codebase",
        )

        # Index data document
        await indexer.index_document(
            content="CSV with columns: id, name, email, status",
            metadata={"type": "data", "format": "csv", "file": "users.csv"},
            doc_id="data_1",
            source_name="data",
        )

        # Index config document
        await indexer.index_document(
            content="Configuration: database_host=localhost, cache_enabled=true",
            metadata={"type": "config", "format": "json", "file": "config.json"},
            doc_id="config_1",
            source_name="config",
        )

        # Test type-based filtering
        code_results = await redis_store.search_similar(
            "process data", k=10, metadata_filter={"type": "code"}
        )
        assert all(r.get("metadata", {}).get("type") == "code" for r in code_results)

        # Test source-based filtering
        data_results = await redis_store.search_similar(
            "user information", k=10, source_filter="data"
        )
        assert all(r.get("source") == "data" for r in data_results)

        # Test combined filters
        config_results = await redis_store.search_similar(
            "configuration settings",
            k=10,
            source_filter="config",
            metadata_filter={"format": "json"},
        )
        assert all(
            r.get("source") == "config" and r.get("metadata", {}).get("format") == "json"
            for r in config_results
        )

    @pytest.mark.asyncio
    async def test_knowledge_graph_performance(self, redis_store, embedding_manager):
        """Test knowledge graph performance with larger dataset."""
        config = MultimodalConfig(
            enable_code_analysis=True,
            enable_pattern_detection=False,  # Disable for performance test
        )

        builder = EnhancedKnowledgeGraphBuilder(
            redis_store.client, redis_store.embedding_manager, config
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create multiple Python files
            for i in range(10):
                file_path = tmpdir_path / f"module_{i}.py"
                file_path.write_text(
                    f"""
class Module{i}:
    def method_{i}_1(self): pass
    def method_{i}_2(self): pass
    def method_{i}_3(self): pass

def function_{i}_1(): pass
def function_{i}_2(): pass
"""
                )

            # Measure build time
            import time

            start_time = time.time()

            stats = await builder.build_from_code(tmpdir_path)

            build_time = time.time() - start_time

            # Performance assertions
            assert build_time < 5.0  # Should complete within 5 seconds
            assert stats["entities"] >= 80  # 10 files * (1 class + 5 functions + module)
            assert stats["files_processed"] == 10

            # Test query performance
            start_time = time.time()

            # Find similar entities (should use embeddings if available)
            similar = await builder.find_similar_entities("Module5", k=5)

            query_time = time.time() - start_time
            assert query_time < 1.0  # Query should be fast

    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self, redis_store, embedding_manager):
        """Test error handling and recovery in the multimodal pipeline."""
        config = MultimodalConfig(enable_code_analysis=True, enable_data_extraction=True)

        builder = EnhancedKnowledgeGraphBuilder(
            redis_store.client, redis_store.embedding_manager, config
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create a malformed Python file
            bad_python = tmpdir_path / "bad_syntax.py"
            bad_python.write_text("def broken_function(: pass")  # Syntax error

            # Create a valid Python file
            good_python = tmpdir_path / "good.py"
            good_python.write_text("def valid_function(): pass")

            # Should handle syntax errors gracefully
            stats = await builder.build_from_code(tmpdir_path)
            assert stats["entities"] > 0  # Should process the good file
            assert stats["files_processed"] >= 1

            # Create invalid JSON
            bad_json = tmpdir_path / "bad.json"
            bad_json.write_text("{invalid json}")

            # Should handle JSON errors gracefully
            data_stats = await builder.build_from_data(bad_json)
            assert data_stats["entities"] == 0  # No entities from bad JSON

            # Create valid CSV
            good_csv = tmpdir_path / "good.csv"
            good_csv.write_text("id,name\\n1,test")

            # Should process valid CSV
            csv_stats = await builder.build_from_data(good_csv)
            assert csv_stats["entities"] > 0
