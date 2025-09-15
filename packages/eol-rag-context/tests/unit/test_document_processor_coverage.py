"""Unit tests to improve document_processor.py coverage."""

import pytest

from eol.rag_context.config import ChunkingConfig, DocumentConfig
from eol.rag_context.document_processor import DocumentProcessor


@pytest.fixture
def chunking_config():
    """Create test chunking configuration."""
    return ChunkingConfig(
        strategy="fixed",
        chunk_size=500,
        chunk_overlap=50,
        semantic_window=3,
        code_block_size=100,
    )


@pytest.fixture
def document_config():
    """Create test document configuration."""
    return DocumentConfig(
        file_patterns=["*.txt", "*.md", "*.py", "*.json", "*.xml", "*.yaml"],
        max_file_size_mb=10,
        extract_metadata=True,
        detect_language=True,
        parse_code_structure=True,
        skip_binary_files=True,
    )


@pytest.fixture
def processor(document_config, chunking_config):
    """Create DocumentProcessor instance."""
    return DocumentProcessor(document_config, chunking_config)


class TestDocumentProcessorCoverage:
    """Tests to improve DocumentProcessor coverage."""

    def test_process_text_file(self, processor, tmp_path):
        """Test processing a text file."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document.\n" * 20)

        # Process the file
        doc_data = processor.process_document(str(test_file))

        # Verify
        assert doc_data is not None
        assert doc_data["source"] == str(test_file)
        assert doc_data["content"]
        assert doc_data["metadata"]["file_type"] == "text"
        assert doc_data["chunks"]

    def test_process_markdown_file(self, processor, tmp_path):
        """Test processing a markdown file."""
        # Create a test markdown file
        test_file = tmp_path / "test.md"
        test_file.write_text(
            """# Header 1

This is some content under header 1.

## Header 2

More content here.

### Header 3

Even more detailed content.
"""
        )

        # Process the file
        doc_data = processor.process_document(str(test_file))

        # Verify
        assert doc_data is not None
        assert doc_data["metadata"]["file_type"] == "markdown"
        assert "chunks" in doc_data
        assert len(doc_data["chunks"]) > 0

    def test_process_python_file(self, processor, tmp_path):
        """Test processing a Python file."""
        # Create a test Python file
        test_file = tmp_path / "test.py"
        test_file.write_text(
            '''"""Module docstring."""

def function1():
    """Function docstring."""
    return "test"

class TestClass:
    """Class docstring."""

    def method(self):
        """Method docstring."""
        pass
'''
        )

        # Process the file
        doc_data = processor.process_document(str(test_file))

        # Verify
        assert doc_data is not None
        assert doc_data["metadata"]["file_type"] == "python"
        assert "chunks" in doc_data

    def test_process_json_file(self, processor, tmp_path):
        """Test processing a JSON file."""
        # Create a test JSON file
        test_file = tmp_path / "test.json"
        test_file.write_text(
            """{
    "name": "test",
    "version": "1.0.0",
    "dependencies": {
        "package1": "1.0.0",
        "package2": "2.0.0"
    },
    "config": {
        "setting1": true,
        "setting2": "value"
    }
}"""
        )

        # Process the file
        doc_data = processor.process_document(str(test_file))

        # Verify
        assert doc_data is not None
        assert doc_data["metadata"]["file_type"] == "json"
        assert "chunks" in doc_data

    def test_process_yaml_file(self, processor, tmp_path):
        """Test processing a YAML file."""
        # Create a test YAML file
        test_file = tmp_path / "test.yaml"
        test_file.write_text(
            """name: test
version: 1.0.0
dependencies:
  package1: 1.0.0
  package2: 2.0.0
config:
  setting1: true
  setting2: value
nested:
  level1:
    level2:
      value: deep
"""
        )

        # Process the file
        doc_data = processor.process_document(str(test_file))

        # Verify
        assert doc_data is not None
        assert doc_data["metadata"]["file_type"] == "yaml"
        assert "chunks" in doc_data

    def test_process_xml_file(self, processor, tmp_path):
        """Test processing an XML file."""
        # Create a test XML file
        test_file = tmp_path / "test.xml"
        test_file.write_text(
            """<?xml version="1.0"?>
<root>
    <section id="1">
        <title>Section 1</title>
        <content>This is section 1 content.</content>
    </section>
    <section id="2">
        <title>Section 2</title>
        <content>This is section 2 content.</content>
    </section>
    <config>
        <setting name="test">value</setting>
    </config>
</root>
"""
        )

        # Process the file
        doc_data = processor.process_document(str(test_file))

        # Verify
        assert doc_data is not None
        assert doc_data["metadata"]["file_type"] == "xml"
        assert "chunks" in doc_data

    def test_process_unsupported_file(self, processor, tmp_path):
        """Test processing an unsupported file type."""
        # Create a test file with unsupported extension
        test_file = tmp_path / "test.xyz"
        test_file.write_text("Some content")

        # Process should return None for unsupported files
        doc_data = processor.process_document(str(test_file))
        assert doc_data is None

    def test_process_large_file(self, processor, tmp_path):
        """Test processing a file that exceeds max size."""
        # Create a large file
        test_file = tmp_path / "large.txt"
        # Write more than max_file_size_mb (convert to bytes)
        max_bytes = processor.document_config.max_file_size_mb * 1024 * 1024
        test_file.write_text("x" * (max_bytes + 1))

        # Process should handle large files
        doc_data = processor.process_document(str(test_file))
        # Should return None or handle gracefully
        assert doc_data is None or "error" in doc_data.get("metadata", {})

    def test_chunk_by_semantic(self, processor):
        """Test semantic chunking strategy."""
        processor.chunking_config.strategy = "semantic"

        text = """This is the first paragraph. It contains some information.

This is the second paragraph. It has different information.

This is the third paragraph. More content here.

This is the fourth paragraph. Even more content.
"""

        chunks = processor._chunk_text_semantic(text, "test.txt")
        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)

    def test_chunk_by_headers(self, processor):
        """Test markdown header-based chunking."""
        text = """# Main Header

Content under main header.

## Sub Header 1

Content under first sub header.

## Sub Header 2

Content under second sub header.

### Sub Sub Header

Nested content here.
"""

        chunks = processor._chunk_by_headers(text, "test.md")
        assert len(chunks) > 0
        assert all("content" in chunk for chunk in chunks)

    def test_extract_metadata(self, processor):
        """Test metadata extraction."""
        # Test with Python file
        python_content = '''"""
Module for testing.

Author: Test Author
Version: 1.0.0
"""

def main():
    pass
'''
        metadata = processor._extract_metadata(python_content, "test.py")
        assert metadata
        assert metadata.get("file_type") == "python"

    def test_process_empty_file(self, processor, tmp_path):
        """Test processing an empty file."""
        # Create an empty file
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        # Process the file
        doc_data = processor.process_document(str(test_file))

        # Should handle empty files gracefully
        assert doc_data is None or doc_data["content"] == ""

    def test_process_binary_file(self, processor, tmp_path):
        """Test processing a binary file."""
        # Create a binary file
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03\x04")

        # Process should handle binary files
        doc_data = processor.process_document(str(test_file))
        assert doc_data is None

    def test_chunking_with_overlap(self, processor):
        """Test chunking with overlap."""
        text = "word " * 200  # Create text with 200 words

        chunks = processor._chunk_text_fixed(text, "test.txt")

        # Check that chunks have overlap
        assert len(chunks) > 1
        if len(chunks) > 1:
            # Check for overlap between consecutive chunks
            chunk1_words = chunks[0]["content"].split()
            chunk2_words = chunks[1]["content"].split()

            # There should be some overlap
            overlap = set(chunk1_words[-processor.chunking_config.chunk_overlap :]) & set(
                chunk2_words[: processor.chunking_config.chunk_overlap]
            )
            assert len(overlap) > 0

    def test_code_ast_chunking(self, processor):
        """Test AST-based code chunking."""
        code = '''
def function1():
    """First function."""
    return 1

def function2():
    """Second function."""
    return 2

class MyClass:
    """A test class."""

    def method1(self):
        return "method1"

    def method2(self):
        return "method2"
'''

        chunks = processor._chunk_code_ast(code, "test.py")
        assert len(chunks) > 0
        # Should have chunks for functions and class
        assert any("function1" in chunk["content"] for chunk in chunks)
        assert any("MyClass" in chunk["content"] for chunk in chunks)
