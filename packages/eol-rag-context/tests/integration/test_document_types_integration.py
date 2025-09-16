"""
Integration tests for document_processor with various file types.
Tests processing of different document formats in real scenarios.
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml


@pytest.mark.integration
class TestDocumentTypesIntegration:
    """Test document processing with various file types."""

    @pytest.mark.asyncio
    async def test_markdown_processing(
        self, document_processor_instance, embedding_manager, redis_store
    ):
        """Test processing of Markdown documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            # Create a comprehensive Markdown file
            md_file = temp_dir / "test_document.md"
            md_file.write_text(
                """# Main Title

## Introduction
This is an introduction paragraph with **bold text** and *italic text*.

### Subsection 1
- List item 1
- List item 2
- List item 3

### Subsection 2
1. Numbered item 1
2. Numbered item 2
3. Numbered item 3

## Code Examples

Here's a Python code block:

```python
def hello_world():
    print("Hello, World!")
    return True
```

And an inline code example: `variable = 42`

## Tables

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |

## Links and References

Check out [this link](https://example.com) for more information.

![Image description](image.png)

## Conclusion

This document demonstrates various Markdown features.

---

Footer information here.
"""
            )

            # Process the document
            doc = await document_processor_instance.process_file(md_file)

            # Verify processing
            assert doc is not None
            assert doc.doc_type == "markdown"
            assert "Main Title" in doc.content
            assert "hello_world" in doc.content
            assert len(doc.chunks) > 0

            # Generate embeddings for chunks
            for chunk in doc.chunks[:3]:  # Test first 3 chunks
                embedding = await embedding_manager.get_embedding(chunk.get("content", ""))
                assert embedding is not None
                assert embedding.shape[0] == 384

    @pytest.mark.asyncio
    async def test_json_processing(
        self, document_processor_instance, embedding_manager, redis_store
    ):
        """Test processing of JSON documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            # Create various JSON files

            # Simple JSON
            simple_json = temp_dir / "simple.json"
            simple_json.write_text(
                json.dumps(
                    {
                        "name": "Test Document",
                        "version": "1.0.0",
                        "description": "A test JSON document for integration testing",
                        "metadata": {
                            "author": "Test Author",
                            "date": "2024-01-15",
                            "tags": ["test", "integration", "json"],
                        },
                    },
                    indent=2,
                )
            )

            # Complex nested JSON
            complex_json = temp_dir / "complex.json"
            complex_json.write_text(
                json.dumps(
                    {
                        "api": {
                            "endpoints": [
                                {"path": "/users", "method": "GET", "description": "Get all users"},
                                {
                                    "path": "/users/{id}",
                                    "method": "GET",
                                    "description": "Get user by ID",
                                },
                            ],
                            "authentication": {"type": "Bearer", "required": True},
                        },
                        "configuration": {
                            "database": {"host": "localhost", "port": 5432, "name": "testdb"},
                            "cache": {"enabled": True, "ttl": 3600},
                        },
                    },
                    indent=2,
                )
            )

            # Process JSON files
            simple_doc = await document_processor_instance.process_file(simple_json)
            complex_doc = await document_processor_instance.process_file(complex_json)

            # Verify simple JSON processing
            assert simple_doc.doc_type == "structured"
            assert "Test Document" in simple_doc.content
            assert "Test Author" in simple_doc.content

            # Verify complex JSON processing
            assert complex_doc.doc_type == "structured"
            assert "/users" in complex_doc.content
            assert "Bearer" in complex_doc.content
            assert "localhost" in complex_doc.content

    @pytest.mark.asyncio
    async def test_yaml_processing(
        self, document_processor_instance, embedding_manager, redis_store
    ):
        """Test processing of YAML documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            # Create YAML configuration file
            yaml_file = temp_dir / "config.yaml"
            yaml_content = {
                "application": {
                    "name": "RAG System",
                    "version": "2.0.0",
                    "environment": "production",
                },
                "services": {
                    "redis": {"host": "redis.example.com", "port": 6379, "password": "secret"},
                    "embedding": {"model": "all-MiniLM-L6-v2", "dimension": 384, "batch_size": 32},
                },
                "features": ["indexing", "searching", "caching", "monitoring"],
            }

            yaml_file.write_text(yaml.dump(yaml_content, default_flow_style=False))

            # Process YAML file
            doc = await document_processor_instance.process_file(yaml_file)

            # Verify processing
            assert doc.doc_type == "structured"
            assert "RAG System" in doc.content
            assert "redis.example.com" in doc.content
            assert "all-MiniLM-L6-v2" in doc.content
            assert len(doc.chunks) > 0

    @pytest.mark.asyncio
    async def test_python_code_processing(
        self, document_processor_instance, embedding_manager, redis_store
    ):
        """Test processing of Python code files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            # Create a Python file with various constructs
            py_file = temp_dir / "sample_code.py"
            py_file.write_text(
                '''"""
Module docstring: This is a sample Python module for testing.
"""

import asyncio
import json
from typing import List, Dict, Optional

# Constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3


class DataProcessor:
    """A class for processing data."""

    def __init__(self, config: Dict):
        """Initialize the processor with configuration."""
        self.config = config
        self.data = []

    def process(self, input_data: List[str]) -> List[Dict]:
        """Process input data and return results."""
        results = []
        for item in input_data:
            processed = self._transform(item)
            results.append(processed)
        return results

    def _transform(self, item: str) -> Dict:
        """Transform a single item."""
        return {
            "original": item,
            "processed": item.upper(),
            "length": len(item)
        }


async def async_function(param: str) -> str:
    """An async function example."""
    await asyncio.sleep(1)
    return f"Processed: {param}"


def main():
    """Main entry point."""
    processor = DataProcessor({"debug": True})
    data = ["hello", "world", "test"]
    results = processor.process(data)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
'''
            )

            # Process Python file
            doc = await document_processor_instance.process_file(py_file)

            # Verify processing
            assert doc.doc_type == "code"
            assert doc.language == "python"
            assert "DataProcessor" in doc.content
            assert "async_function" in doc.content
            assert "process" in doc.content

            # Check code structure extraction
            assert doc.metadata is not None
            if "classes" in doc.metadata:
                assert "DataProcessor" in doc.metadata["classes"]
            if "functions" in doc.metadata:
                assert "main" in doc.metadata["functions"]

    @pytest.mark.asyncio
    async def test_javascript_processing(
        self, document_processor_instance, embedding_manager, redis_store
    ):
        """Test processing of JavaScript/TypeScript files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            # Create JavaScript file
            js_file = temp_dir / "app.js"
            js_file.write_text(
                """// Main application file

const express = require('express');
const app = express();

// Configuration
const config = {
    port: 3000,
    host: 'localhost',
    debug: true
};

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.get('/', (req, res) => {
    res.json({ message: 'Hello World' });
});

app.post('/data', async (req, res) => {
    try {
        const { data } = req.body;
        const processed = await processData(data);
        res.json({ result: processed });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Helper functions
async function processData(data) {
    // Simulate async processing
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve(data.toUpperCase());
        }, 100);
    });
}

// Start server
app.listen(config.port, () => {
    console.log(`Server running at http://${config.host}:${config.port}`);
});

module.exports = app;
"""
            )

            # Process JavaScript file
            doc = await document_processor_instance.process_file(js_file)

            # Verify processing
            assert doc.doc_type == "code"
            assert doc.language == "javascript"
            assert "express" in doc.content
            assert "processData" in doc.content
            assert "app.listen" in doc.content

    @pytest.mark.asyncio
    async def test_xml_processing(
        self, document_processor_instance, embedding_manager, redis_store
    ):
        """Test processing of XML documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            # Create XML file
            xml_file = temp_dir / "data.xml"
            xml_file.write_text(
                """<?xml version="1.0" encoding="UTF-8"?>
<catalog>
    <product id="1">
        <name>Product One</name>
        <description>This is the first product</description>
        <price currency="USD">29.99</price>
        <category>Electronics</category>
        <tags>
            <tag>gadget</tag>
            <tag>popular</tag>
        </tags>
    </product>
    <product id="2">
        <name>Product Two</name>
        <description>This is the second product</description>
        <price currency="EUR">39.99</price>
        <category>Books</category>
        <tags>
            <tag>education</tag>
            <tag>bestseller</tag>
        </tags>
    </product>
</catalog>"""
            )

            # Process XML file
            doc = await document_processor_instance.process_file(xml_file)

            # Verify processing
            assert doc.doc_type == "xml"
            assert "Product One" in doc.content
            assert "Product Two" in doc.content
            assert "Electronics" in doc.content
            assert doc.metadata["element_count"] > 0

    @pytest.mark.asyncio
    async def test_csv_processing(
        self, document_processor_instance, embedding_manager, redis_store
    ):
        """Test processing of CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            # Create CSV file
            csv_file = temp_dir / "data.csv"
            csv_file.write_text(
                """Name,Age,Department,Salary,Location
John Doe,30,Engineering,75000,New York
Jane Smith,28,Marketing,65000,San Francisco
Bob Johnson,35,Sales,70000,Chicago
Alice Williams,32,Engineering,80000,Boston
Charlie Brown,29,HR,60000,Seattle
Diana Prince,31,Marketing,68000,Los Angeles"""
            )

            # Process CSV file
            doc = await document_processor_instance.process_file(csv_file)

            # Verify processing
            assert doc.doc_type in ["csv", "text"]  # Might be detected as text
            assert "John Doe" in doc.content
            assert "Engineering" in doc.content
            assert "75000" in doc.content

    @pytest.mark.asyncio
    async def test_mixed_content_directory(
        self, document_processor_instance, embedding_manager, redis_store, server_instance
    ):
        """Test processing a directory with mixed file types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            # Create various file types
            (temp_dir / "readme.md").write_text("# Project README\n\nThis is a test project.")
            (temp_dir / "config.json").write_text('{"version": "1.0", "enabled": true}')
            (temp_dir / "script.py").write_text("def main():\n    print('Hello')")
            (temp_dir / "data.yaml").write_text("key: value\nlist:\n  - item1\n  - item2")
            (temp_dir / "notes.txt").write_text("Plain text notes file")

            # Index the entire directory
            from eol.rag_context.config import RAGConfig
            from eol.rag_context.indexer import DocumentIndexer

            indexer = DocumentIndexer(
                config=RAGConfig(),
                document_processor=document_processor_instance,
                embedding_manager=embedding_manager,
                redis_store=redis_store,
            )

            result = await indexer.index_folder(temp_dir, recursive=False)

            # Verify mixed content processing
            assert result.file_count == 5
            assert result.total_chunks > 0

            # Search for content from different file types
            query_embedding = await embedding_manager.get_embedding("Project README config version")
            search_results = await redis_store.vector_search(query_embedding=query_embedding, k=10)

            # Should find content from multiple file types
            assert len(search_results) > 0

    @pytest.mark.asyncio
    async def test_large_document_chunking(self, document_processor_instance, embedding_manager):
        """Test chunking of large documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            # Create a large document
            large_file = temp_dir / "large_document.md"
            content = "# Large Document\n\n"

            # Add 50 sections
            for i in range(50):
                content += f"## Section {i}\n\n"
                content += f"This is section {i} with substantial content. " * 20
                content += "\n\n"

            large_file.write_text(content)

            # Process with different chunk sizes
            from eol.rag_context.config import ChunkingConfig

            # Small chunks
            document_processor_instance.chunk_config = ChunkingConfig(
                max_chunk_size=200, chunk_overlap=50
            )

            doc_small = await document_processor_instance.process_file(large_file)

            # Large chunks
            document_processor_instance.chunk_config = ChunkingConfig(
                max_chunk_size=1000, chunk_overlap=100
            )

            doc_large = await document_processor_instance.process_file(large_file)

            # Verify chunking
            assert len(doc_small.chunks) > len(doc_large.chunks)
            assert all(
                len(chunk.get("content", "")) <= 1200 for chunk in doc_small.chunks
            )  # With overlap
            assert all(
                len(chunk.get("content", "")) <= 2000 for chunk in doc_large.chunks
            )  # With overlap

    @pytest.mark.asyncio
    async def test_text_file_processing(self, document_processor_instance):
        """Test processing of plain text files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            # Create a text file
            text_file = temp_dir / "document.txt"
            text_file.write_text(
                """This is a plain text document.
It has multiple lines of content.
Each line contains different information.

This is a new paragraph after a blank line.
It continues with more text content.
The document has various topics and sections.

Final paragraph with concluding remarks.
This ensures we have enough content for chunking."""
            )

            # Process the text file
            doc = await document_processor_instance.process_file(text_file)

            # Verify processing
            assert doc is not None
            assert doc.doc_type == "text"
            assert "plain text document" in doc.content
            assert "multiple lines" in doc.content
            assert len(doc.chunks) > 0

    @pytest.mark.asyncio
    async def test_empty_file_handling(self, document_processor_instance):
        """Test handling of empty files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            # Create an empty file
            empty_file = temp_dir / "empty.txt"
            empty_file.write_text("")

            # Process the empty file
            doc = await document_processor_instance.process_file(empty_file)

            # Should handle gracefully - empty files may return None or empty document
            if doc is None:
                # Acceptable to return None for empty files
                pass
            else:
                assert doc.content == ""
                assert len(doc.chunks) == 0
