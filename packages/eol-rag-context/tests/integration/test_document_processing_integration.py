"""Integration tests for document processing.

Tests real file processing with various formats.

"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from eol.rag_context import document_processor


@pytest.mark.integration
class TestDocumentProcessingIntegration:
    """Test document processing with real files."""

    @pytest.mark.asyncio
    async def test_process_text_file(self, document_processor_instance):
        """Test processing plain text files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            content = """This is a test document.
It contains multiple lines of text.
Each line has some information.

This is a new paragraph after a blank line.
It should be properly processed and chunked."""
            f.write(content)
            f.flush()

            try:
                doc = await document_processor_instance.process_file(Path(f.name))

                assert doc is not None
                assert doc.doc_type == "text"
                assert doc.content == content
                assert len(doc.chunks) > 0

                # Verify chunks
                for chunk in doc.chunks:
                    assert "content" in chunk
                    assert "metadata" in chunk
                    assert chunk["content"] != ""
            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_process_markdown_file(self, document_processor_instance):
        """Test processing Markdown files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            content = """# Main Title

This is the introduction paragraph with **bold** and *italic* text.

## Section 1
Content for section 1 with some details.

### Subsection 1.1
More specific information here.

## Section 2
Another major section with content.

```python
def example():
    return "code block"
```

- List item 1
- List item 2
- List item 3
"""
            f.write(content)
            f.flush()

            try:
                doc = await document_processor_instance.process_file(Path(f.name))

                assert doc is not None
                assert doc.doc_type == "markdown"
                assert "Main Title" in doc.content
                assert len(doc.chunks) > 0

                # Check for proper markdown processing
                headers_found = False
                for chunk in doc.chunks:
                    if "metadata" in chunk and "header" in chunk.get("metadata", {}):
                        headers_found = True
                        break
                assert headers_found
            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_process_python_file(self, document_processor_instance):
        """Test processing Python code files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            content = '''"""Module docstring."""

import os
import sys

class MyClass:
    """A sample class."""

    def __init__(self, value):
        """Initialize the class."""
        self.value = value

    def get_value(self):
        """Get the value."""
        return self.value

    @property
    def double_value(self):
        """Return double the value."""
        return self.value * 2

def standalone_function(param1, param2):
    """A standalone function."""
    if param1 > param2:
        return param1
    return param2

# Global constant
CONSTANT = 42

if __name__ == "__main__":
    obj = MyClass(10)
    print(obj.get_value())
'''
            f.write(content)
            f.flush()

            try:
                doc = await document_processor_instance.process_file(Path(f.name))

                assert doc is not None
                assert doc.doc_type == "code"
                assert "MyClass" in doc.content
                assert "standalone_function" in doc.content
                assert len(doc.chunks) > 0

                # Verify code chunks
                for chunk in doc.chunks:
                    assert "content" in chunk
                    metadata = chunk.get("metadata", {})
                    assert "language" in metadata
                    assert metadata["language"] == "python"
            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_process_json_file(self, document_processor_instance):
        """Test processing JSON files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            data = {
                "name": "test-project",
                "version": "1.0.0",
                "dependencies": {"package1": "^1.0.0", "package2": "~2.0.0"},
                "config": {
                    "setting1": True,
                    "setting2": 42,
                    "nested": {"deep": "value"},
                },
                "items": ["item1", "item2", "item3"],
            }
            json.dump(data, f, indent=2)
            f.flush()

            try:
                doc = await document_processor_instance.process_file(Path(f.name))

                assert doc is not None
                assert doc.doc_type == "structured"
                assert "test-project" in doc.content
                assert len(doc.chunks) > 0

                # Verify structured data chunks
                for chunk in doc.chunks:
                    assert "content" in chunk
                    assert "metadata" in chunk
                    metadata = chunk.get("metadata", {})
                    assert (
                        metadata.get("format") == "structured"
                    )  # JSON files are processed as structured data
            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_process_javascript_file(self, document_processor_instance):
        """Test processing JavaScript files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            content = """// JavaScript test file

class Calculator {
    constructor() {
        this.result = 0;
    }

    add(a, b) {
        return a + b;
    }

    multiply(a, b) {
        return a * b;
    }
}

function processData(data) {
    return data.map(item => item * 2);
}

const CONFIG = {
    apiUrl: 'https://api.example.com',
    timeout: 5000
};

export { Calculator, processData, CONFIG };
"""
            f.write(content)
            f.flush()

            try:
                doc = await document_processor_instance.process_file(Path(f.name))

                assert doc is not None
                assert doc.doc_type == "code"
                assert "Calculator" in doc.content
                assert "processData" in doc.content
                assert len(doc.chunks) > 0

                # Verify language detection
                for chunk in doc.chunks:
                    metadata = chunk.get("metadata", {})
                    assert metadata.get("language") == "javascript"
            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_chunking_strategies(self, document_processor_instance):
        """Test different chunking strategies."""
        # Test with large text to ensure chunking
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            # Generate large content
            content = "\n".join([f"Line {i}: " + "word " * 20 for i in range(100)])
            f.write(content)
            f.flush()

            try:
                doc = await document_processor_instance.process_file(Path(f.name))

                assert doc is not None
                assert len(doc.chunks) > 1  # Should be chunked

                # Verify chunk sizes
                for chunk in doc.chunks:
                    assert len(chunk["content"]) > 0
                    assert (
                        len(chunk["content"])
                        <= document_processor_instance.chunk_config.max_chunk_size
                    )
            finally:
                Path(f.name).unlink()

    @pytest.mark.asyncio
    async def test_language_detection(self, document_processor_instance):
        """Test programming language detection."""
        test_cases = [
            (".py", "python"),
            (".js", "javascript"),
            (".ts", "typescript"),
            (".go", "go"),
            (".rs", "rust"),
            (".java", "java"),
            (".cpp", "cpp"),
            (".rb", "ruby"),
            (".php", "php"),
        ]

        for ext, expected_lang in test_cases:
            lang = document_processor_instance._detect_language(ext)
            assert (
                lang == expected_lang
            ), f"Failed for {ext}: expected {expected_lang}, got {lang}"

    @pytest.mark.asyncio
    async def test_concurrent_processing(
        self, document_processor_instance, temp_test_directory
    ):
        """Test concurrent processing of multiple files."""
        import asyncio

        # Get all files from temp directory
        files = list(temp_test_directory.glob("**/*"))
        files = [f for f in files if f.is_file()]

        # Process all files concurrently
        tasks = [document_processor_instance.process_file(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify results
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) > 0

        for doc in successful:
            if doc:  # Some files might not be supported
                assert doc.content != ""
                assert doc.doc_type in [
                    "text",
                    "code",
                    "markdown",
                    "structured",
                    "html",
                    "pdf",
                    "docx",
                ]

    @pytest.mark.asyncio
    async def test_error_handling(self, document_processor_instance):
        """Test error handling for invalid files."""
        # Test non-existent file
        result = await document_processor_instance.process_file(
            Path("/nonexistent/file.txt")
        )
        assert result is None

        # Test empty file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.flush()  # Empty file

            try:
                doc = await document_processor_instance.process_file(Path(f.name))
                # Should handle gracefully
                if doc:
                    assert doc.content == ""
            finally:
                Path(f.name).unlink()
