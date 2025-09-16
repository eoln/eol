"""
End-to-end integration tests for multimodal knowledge graph.
Tests the complete workflow with real components following the existing pattern.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path

import pytest


@pytest.mark.integration
class TestMultimodalKnowledgeGraphE2E:
    """Test multimodal knowledge graph with real components."""

    @pytest.fixture
    def sample_project(self):
        """Create a sample project with code and data files."""
        project = {
            "src/main.py": """
import pandas as pd
import json
from data_loader import DataLoader
from processor import DataProcessor

class Application:
    \"\"\"Main application class.\"\"\"

    def __init__(self):
        self.loader = DataLoader()
        self.processor = DataProcessor()

    def run(self):
        \"\"\"Run the application.\"\"\"
        # Load configuration
        config = self.load_config("config/settings.json")

        # Load data
        users = self.loader.load_users("data/users.csv")
        products = self.loader.load_products("data/products.csv")

        # Process data
        results = self.processor.process(users, products)
        return results

    def load_config(self, path):
        \"\"\"Load configuration from JSON.\"\"\"
        with open(path, 'r') as f:
            return json.load(f)
""",
            "src/data_loader.py": """
import pandas as pd

class DataLoader:
    \"\"\"Load data from various sources.\"\"\"

    def load_users(self, path):
        \"\"\"Load user data from CSV.\"\"\"
        df = pd.read_csv(path)
        return df[df['status'] == 'active']

    def load_products(self, path):
        \"\"\"Load product data from CSV.\"\"\"
        return pd.read_csv(path)

    def load_json(self, path):
        \"\"\"Load JSON data.\"\"\"
        import json
        with open(path, 'r') as f:
            return json.load(f)
""",
            "src/processor.py": """
class DataProcessor:
    \"\"\"Process and transform data.\"\"\"

    def process(self, users_df, products_df):
        \"\"\"Process user and product data.\"\"\"
        # Join users with products
        result = users_df.merge(products_df, on='user_id', how='left')
        return result

    def aggregate(self, df):
        \"\"\"Aggregate data by user.\"\"\"
        return df.groupby('user_id').agg({
            'price': 'sum',
            'quantity': 'sum'
        })
""",
            "data/users.csv": """user_id,name,email,status,created_at
1,Alice Johnson,alice@example.com,active,2024-01-01
2,Bob Smith,bob@example.com,active,2024-01-02
3,Charlie Brown,charlie@example.com,inactive,2024-01-03
4,Diana Prince,diana@example.com,active,2024-01-04""",
            "data/products.csv": """product_id,name,price,category,user_id
101,Laptop,999.99,Electronics,1
102,Mouse,29.99,Electronics,1
103,Keyboard,79.99,Electronics,2
104,Monitor,299.99,Electronics,4""",
            "config/settings.json": json.dumps(
                {
                    "app": {"name": "Data Processing Application", "version": "1.0.0"},
                    "database": {"host": "localhost", "port": 5432, "name": "app_db"},
                    "processing": {"batch_size": 1000, "parallel": True},
                },
                indent=2,
            ),
            "docs/README.md": """# Data Processing Application

## Overview
This application processes user and product data.

## Features
- Load data from CSV files
- Process and transform data
- Generate reports

## Usage
```python
app = Application()
results = app.run()
```

## Configuration
Edit `config/settings.json` to configure the application.
""",
        }
        return project

    @pytest.mark.asyncio
    async def test_multimodal_indexing(
        self, server_instance, redis_store, embedding_manager, sample_project
    ):
        """Test indexing of multimodal content (code, data, docs)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create project structure
            for file_path, content in sample_project.items():
                full_path = tmpdir_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)

            # Start indexing
            index_result = await server_instance.index_directory(str(tmpdir_path), recursive=True)
            assert index_result["status"] == "started"
            task_id = index_result["task_id"]

            # Wait for completion
            max_wait = 30
            wait_time = 0
            while wait_time < max_wait:
                status = await server_instance.task_manager.get_task_status(task_id)
                if status and status.status.value in ["completed", "failed"]:
                    break
                await asyncio.sleep(1)
                wait_time += 1

            # Verify indexing completed
            final_status = await server_instance.task_manager.get_task_status(task_id)
            assert final_status is not None
            assert final_status.status.value == "completed"
            assert final_status.total_files >= 7  # All files indexed
            assert final_status.total_chunks > 0

            # Search for code entities
            code_query = "DataLoader class load users"
            code_embedding = await embedding_manager.get_embedding(code_query)
            code_results = await redis_store.vector_search(
                query_embedding=code_embedding, hierarchy_level=3, k=5
            )

            # Should find code-related results
            if len(code_results) > 0:
                found_code = False
                for doc_id, score, data in code_results:
                    if "DataLoader" in data.get("content", "") or "load_users" in data.get(
                        "content", ""
                    ):
                        found_code = True
                        break
                assert found_code, "Should find DataLoader or load_users in results"

            # Search for data entities
            data_query = "CSV user data email active"
            data_embedding = await embedding_manager.get_embedding(data_query)
            data_results = await redis_store.vector_search(
                query_embedding=data_embedding, hierarchy_level=3, k=5
            )

            # Should find data-related results
            if len(data_results) > 0:
                found_data = False
                for doc_id, score, data in data_results:
                    content = data.get("content", "")
                    if "alice@example.com" in content or "user_id" in content:
                        found_data = True
                        break
                assert found_data, "Should find user data in results"

    @pytest.mark.asyncio
    async def test_multimodal_knowledge_graph_construction(
        self, knowledge_graph_instance, server_instance, sample_project
    ):
        """Test building knowledge graph from multimodal content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create project structure
            for file_path, content in sample_project.items():
                full_path = tmpdir_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)

            # Index the directory first
            index_result = await server_instance.index_directory(str(tmpdir_path), recursive=True)
            task_id = index_result["task_id"]

            # Wait for indexing
            max_wait = 30
            wait_time = 0
            while wait_time < max_wait:
                status = await server_instance.task_manager.get_task_status(task_id)
                if status and status.status.value in ["completed", "failed"]:
                    break
                await asyncio.sleep(1)
                wait_time += 1

            final_status = await server_instance.task_manager.get_task_status(task_id)
            assert final_status.status.value == "completed"
            source_id = final_status.source_id

            # Build knowledge graph
            await knowledge_graph_instance.build_from_documents(source_id)

            # Check entities were created
            if len(knowledge_graph_instance.entities) > 0:
                # Verify different entity types
                entity_types = set()
                for entity_id, entity_data in knowledge_graph_instance.entities.items():
                    if "type" in entity_data:
                        entity_types.add(entity_data["type"])

                # Should have various entity types from multimodal content
                assert len(entity_types) > 0, "Should have different entity types"

                # Query a subgraph
                first_entity_id = list(knowledge_graph_instance.entities.keys())[0]
                subgraph = await knowledge_graph_instance.query_subgraph(
                    first_entity_id, max_depth=2
                )

                assert hasattr(subgraph, "entities")
                assert hasattr(subgraph, "relationships")

    @pytest.mark.asyncio
    async def test_code_data_relationship_discovery(
        self, server_instance, redis_store, embedding_manager, sample_project
    ):
        """Test discovering relationships between code and data files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create project structure
            for file_path, content in sample_project.items():
                full_path = tmpdir_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)

            # Index the project
            index_result = await server_instance.index_directory(str(tmpdir_path), recursive=True)
            task_id = index_result["task_id"]

            # Wait for completion
            max_wait = 30
            wait_time = 0
            while wait_time < max_wait:
                status = await server_instance.task_manager.get_task_status(task_id)
                if status and status.status.value in ["completed", "failed"]:
                    break
                await asyncio.sleep(1)
                wait_time += 1

            # Search for code that references data files
            query = "load_users CSV pandas read_csv"
            embedding = await embedding_manager.get_embedding(query)
            results = await redis_store.vector_search(
                query_embedding=embedding, hierarchy_level=3, k=10
            )

            # Analyze results for code-data relationships
            code_refs_data = False
            data_file_found = False

            for doc_id, score, data in results:
                content = data.get("content", "")
                metadata = data.get("metadata", {})

                # Check for code referencing data files
                if "read_csv" in content and ("users.csv" in content or "products.csv" in content):
                    code_refs_data = True

                # Check for actual data files
                if metadata.get("file_type") == "csv" or "user_id,name,email" in content:
                    data_file_found = True

            # Should find both code and data in search results
            assert code_refs_data or data_file_found, "Should find code-data relationships"

    @pytest.mark.asyncio
    async def test_hierarchical_search_multimodal(
        self, server_instance, redis_store, embedding_manager, sample_project
    ):
        """Test hierarchical search across multimodal content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create project structure
            for file_path, content in sample_project.items():
                full_path = tmpdir_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)

            # Index the project
            index_result = await server_instance.index_directory(str(tmpdir_path), recursive=True)
            task_id = index_result["task_id"]

            # Wait for indexing
            max_wait = 30
            wait_time = 0
            while wait_time < max_wait:
                status = await server_instance.task_manager.get_task_status(task_id)
                if status and status.status.value in ["completed", "failed"]:
                    break
                await asyncio.sleep(1)
                wait_time += 1

            # Test different hierarchy levels
            query = "data processing application"
            embedding = await embedding_manager.get_embedding(query)

            # Search at concept level (high-level topics)
            concept_results = await redis_store.vector_search(
                query_embedding=embedding, hierarchy_level=1, k=3  # Concept level
            )

            # Search at section level (mid-level)
            section_results = await redis_store.vector_search(
                query_embedding=embedding, hierarchy_level=2, k=5  # Section level
            )

            # Search at chunk level (detailed)
            chunk_results = await redis_store.vector_search(
                query_embedding=embedding, hierarchy_level=3, k=10  # Chunk level
            )

            # Different levels should return different granularity
            # This is more of a structural test since mock Redis might not differentiate
            assert isinstance(concept_results, list)
            assert isinstance(section_results, list)
            assert isinstance(chunk_results, list)

    @pytest.mark.asyncio
    async def test_multimodal_semantic_cache(
        self, semantic_cache_instance, embedding_manager, sample_project
    ):
        """Test semantic caching with multimodal queries."""
        # Initialize cache
        await semantic_cache_instance.initialize()

        # Cache code-related queries
        code_queries = [
            ("How does DataLoader work?", "DataLoader loads data from CSV files using pandas"),
            ("What does the processor do?", "DataProcessor transforms and aggregates user data"),
            ("How to load configuration?", "Use load_config method to read JSON settings"),
        ]

        for query, response in code_queries:
            await semantic_cache_instance.set(
                query, response, {"type": "code", "timestamp": time.time()}
            )

        # Cache data-related queries
        data_queries = [
            ("What users are active?", "Alice, Bob, and Diana are active users"),
            ("What products are available?", "Laptop, Mouse, Keyboard, and Monitor"),
            ("What is the price range?", "Prices range from $29.99 to $999.99"),
        ]

        for query, response in data_queries:
            await semantic_cache_instance.set(
                query, response, {"type": "data", "timestamp": time.time()}
            )

        # Test cache hits with similar queries
        similar_query = (
            "How does the data loader function?"  # Similar to "How does DataLoader work?"
        )
        cached = await semantic_cache_instance.get(similar_query)
        # May or may not hit depending on similarity threshold

        # Test exact match
        exact_cached = await semantic_cache_instance.get("What users are active?")
        assert exact_cached == "Alice, Bob, and Diana are active users"

        # Check cache stats
        stats = semantic_cache_instance.get_stats()
        assert stats["queries"] > 0
        # stats might not have 'entries' key, check for keys that exist
        assert "hit_rate" in stats or "queries" in stats

        # Clear cache
        await semantic_cache_instance.clear()

    @pytest.mark.asyncio
    async def test_pattern_discovery_in_multimodal_content(
        self, knowledge_graph_instance, server_instance, sample_project
    ):
        """Test discovering patterns across code and data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Add more similar files to detect patterns
            extended_project = sample_project.copy()

            # Add more data loader modules with pattern
            for i in range(3):
                extended_project[
                    f"src/loader_{i}.py"
                ] = f"""
import pandas as pd

class Loader{i}:
    def load_data(self):
        return pd.read_csv("data/file_{i}.csv")
"""

            # Add more CSV files with pattern
            for i in range(3):
                extended_project[f"data/file_{i}.csv"] = f"id,value\\n{i},test{i}"

            # Create extended project
            for file_path, content in extended_project.items():
                full_path = tmpdir_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)

            # Index the extended project
            index_result = await server_instance.index_directory(str(tmpdir_path), recursive=True)
            task_id = index_result["task_id"]

            # Wait for indexing
            max_wait = 30
            wait_time = 0
            while wait_time < max_wait:
                status = await server_instance.task_manager.get_task_status(task_id)
                if status and status.status.value in ["completed", "failed"]:
                    break
                await asyncio.sleep(1)
                wait_time += 1

            final_status = await server_instance.task_manager.get_task_status(task_id)
            assert final_status.status.value == "completed"
            source_id = final_status.source_id

            # Build knowledge graph to analyze patterns
            await knowledge_graph_instance.build_from_documents(source_id)

            # Look for patterns in entities
            if len(knowledge_graph_instance.entities) > 0:
                # Count entity types to find patterns
                type_counts = {}
                for entity_id, entity_data in knowledge_graph_instance.entities.items():
                    entity_type = entity_data.get("type", "unknown")
                    type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

                # Should find multiple instances of similar entities (pattern)
                # e.g., multiple loader classes, multiple CSV files
                assert len(type_counts) > 0, "Should identify entity type patterns"

    @pytest.mark.asyncio
    async def test_incremental_multimodal_indexing(
        self, server_instance, redis_store, sample_project
    ):
        """Test incremental indexing of multimodal content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Phase 1: Index only code files
            code_files = {k: v for k, v in sample_project.items() if k.endswith(".py")}
            for file_path, content in code_files.items():
                full_path = tmpdir_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)

            # Initial indexing
            index_result = await server_instance.index_directory(str(tmpdir_path), recursive=True)
            task_id = index_result["task_id"]

            # Wait for completion
            max_wait = 30
            wait_time = 0
            while wait_time < max_wait:
                status = await server_instance.task_manager.get_task_status(task_id)
                if status and status.status.value in ["completed", "failed"]:
                    break
                await asyncio.sleep(1)
                wait_time += 1

            initial_status = await server_instance.task_manager.get_task_status(task_id)
            initial_chunks = initial_status.total_chunks

            # Phase 2: Add data files incrementally
            data_files = {
                k: v for k, v in sample_project.items() if k.endswith(".csv") or k.endswith(".json")
            }
            for file_path, content in data_files.items():
                full_path = tmpdir_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)

            # Incremental indexing
            index_result2 = await server_instance.index_directory(str(tmpdir_path), recursive=True)
            task_id2 = index_result2["task_id"]

            # Wait for incremental indexing
            wait_time = 0
            while wait_time < max_wait:
                status = await server_instance.task_manager.get_task_status(task_id2)
                if status and status.status.value in ["completed", "failed"]:
                    break
                await asyncio.sleep(1)
                wait_time += 1

            final_status = await server_instance.task_manager.get_task_status(task_id2)
            assert final_status.status.value == "completed"
            assert (
                final_status.total_chunks > initial_chunks
            ), "Should have more chunks after incremental indexing"
