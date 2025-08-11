"""
Improved tests for document_processor to boost coverage from 52% to 70%.
Targets specific uncovered lines.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

# Mock all external dependencies
for module in [
    "magic",
    "pypdf",
    "docx",
    "aiofiles",
    "aiofiles.os",
    "tree_sitter",
    "tree_sitter_python",
    "tree_sitter_javascript",
    "tree_sitter_typescript",
    "tree_sitter_go",
    "tree_sitter_rust",
    "tree_sitter_cpp",
    "tree_sitter_c",
    "tree_sitter_java",
    "tree_sitter_csharp",
    "tree_sitter_ruby",
    "tree_sitter_php",
    "yaml",
    "bs4",
    "markdown",
]:
    sys.modules[module] = MagicMock()

from eol.rag_context import config, document_processor


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_initialization_and_tree_sitter():
    """Test initialization including tree-sitter setup (lines 23-26)."""
    proc = document_processor.DocumentProcessor(config.DocumentConfig(), config.ChunkingConfig())
    assert proc.doc_config is not None
    assert proc.chunk_config is not None


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_process_text_file():
    """Test _process_text method (lines 77-95)."""
    proc = document_processor.DocumentProcessor(config.DocumentConfig(), config.ChunkingConfig())

    with patch("eol.rag_context.document_processor.aiofiles.open") as mock_aio:
        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value="This is test content.\nLine 2.\nLine 3.")
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock()
        mock_aio.return_value = mock_file

        doc = await proc._process_text(Path("/test.txt"))
        assert doc.doc_type == "text"
        assert doc.content == "This is test content.\nLine 2.\nLine 3."
        assert len(doc.chunks) > 0


@pytest.mark.asyncio
async def test_process_markdown_file():
    """Test _process_markdown method (lines 103-132)."""
    proc = document_processor.DocumentProcessor(config.DocumentConfig(), config.ChunkingConfig())

    markdown_content = """# Main Title

This is the introduction paragraph.

## Section 1
Content for section 1.

### Subsection 1.1
Details in subsection.

## Section 2
Content for section 2.

```python
def example():
    return "code"
```

- List item 1
- List item 2
"""

    with patch("eol.rag_context.document_processor.aiofiles.open") as mock_aio:
        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=markdown_content)
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock()
        mock_aio.return_value = mock_file

        # Mock markdown processor
        with patch("eol.rag_context.document_processor.markdown") as mock_md:
            mock_md.markdown = MagicMock(return_value="<h1>Main Title</h1><p>Content</p>")

            doc = await proc._process_markdown(Path("/test.md"))
            assert doc.doc_type == "markdown"
            assert "Main Title" in doc.content
            assert len(doc.chunks) > 0


@pytest.mark.asyncio
async def test_process_pdf_file():
    """Test _process_pdf method (lines 211-242)."""
    proc = document_processor.DocumentProcessor(config.DocumentConfig(), config.ChunkingConfig())

    with (
        patch("builtins.open", mock_open(read_data=b"PDF content")),
        patch("eol.rag_context.document_processor.pypdf.PdfReader") as MockPdf,
    ):

        mock_reader = MagicMock()

        # Create multiple pages with different content
        pages = []
        for i in range(5):
            mock_page = MagicMock()
            mock_page.extract_text.return_value = (
                f"Page {i+1} content.\nWith multiple lines.\nAnd paragraphs."
            )
            pages.append(mock_page)

        mock_reader.pages = pages
        MockPdf.return_value = mock_reader

        doc = await proc._process_pdf(Path("/test.pdf"))
        assert doc.doc_type == "pdf"
        assert "Page 1 content" in doc.content
        assert "Page 5 content" in doc.content
        assert len(doc.chunks) > 0


@pytest.mark.asyncio
async def test_process_docx_file():
    """Test _process_docx method (lines 264-304)."""
    proc = document_processor.DocumentProcessor(config.DocumentConfig(), config.ChunkingConfig())

    with patch("eol.rag_context.document_processor.docx.Document") as MockDocx:
        mock_doc = MagicMock()

        # Create paragraphs
        paragraphs = []
        for i in range(10):
            mock_para = MagicMock()
            mock_para.text = f"Paragraph {i+1} with some content."
            paragraphs.append(mock_para)
        mock_doc.paragraphs = paragraphs

        # Create tables
        mock_table = MagicMock()
        mock_row1 = MagicMock()
        mock_row2 = MagicMock()

        mock_cell1 = MagicMock()
        mock_cell1.text = "Header 1"
        mock_cell2 = MagicMock()
        mock_cell2.text = "Header 2"
        mock_row1.cells = [mock_cell1, mock_cell2]

        mock_cell3 = MagicMock()
        mock_cell3.text = "Data 1"
        mock_cell4 = MagicMock()
        mock_cell4.text = "Data 2"
        mock_row2.cells = [mock_cell3, mock_cell4]

        mock_table.rows = [mock_row1, mock_row2]
        mock_doc.tables = [mock_table]

        MockDocx.return_value = mock_doc

        doc = await proc._process_docx(Path("/test.docx"))
        assert doc.doc_type == "docx"
        assert "Paragraph 1" in doc.content
        assert "Header 1" in doc.content
        assert "Data 1" in doc.content
        assert len(doc.chunks) > 0


@pytest.mark.asyncio
async def test_process_html_file():
    """Test _process_html method (lines 159, 167)."""
    proc = document_processor.DocumentProcessor(config.DocumentConfig(), config.ChunkingConfig())

    html_content = """<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
    <h1>Main Title</h1>
    <h2>Subtitle</h2>
    <p>This is a paragraph with <strong>bold</strong> text.</p>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
    </ul>
    <div>
        <h3>Section</h3>
        <p>Another paragraph.</p>
    </div>
</body>
</html>"""

    with patch("eol.rag_context.document_processor.aiofiles.open") as mock_aio:
        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value=html_content)
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock()
        mock_aio.return_value = mock_file

        doc = await proc._process_html(Path("/test.html"))
        assert doc.doc_type == "html"
        assert len(doc.chunks) > 0


@pytest.mark.asyncio
async def test_chunk_markdown_by_headers():
    """Test _chunk_markdown_by_headers method (lines 357-396)."""
    proc = document_processor.DocumentProcessor(
        config.DocumentConfig(),
        config.ChunkingConfig(max_chunk_size=100),  # Small size to force chunking
    )

    markdown_text = """# Header 1
Content under header 1.

## Header 2
Content under header 2 with more text to make it longer.
This continues for a while to test chunking.

### Header 3
Nested content under header 3.

## Another Header 2
More content here.

# Header 1 Again
Final section content."""

    chunks = proc._chunk_markdown_by_headers(markdown_text)
    assert len(chunks) > 0
    assert any("header" in chunk.get("metadata", {}) for chunk in chunks)

    # Test with very long content under one header
    long_markdown = "# Single Header\n" + ("Long content line. " * 100)
    chunks = proc._chunk_markdown_by_headers(long_markdown)
    assert len(chunks) > 1  # Should be split due to size


@pytest.mark.asyncio
async def test_chunk_code_by_ast():
    """Test _chunk_code_by_ast method (lines 421-450)."""
    proc = document_processor.DocumentProcessor(config.DocumentConfig(), config.ChunkingConfig())

    python_code = """
def function1():
    '''Docstring for function1'''
    x = 1
    y = 2
    return x + y

class MyClass:
    '''Class docstring'''
    
    def __init__(self):
        self.value = 0
    
    def method1(self):
        return self.value
    
    @property
    def prop(self):
        return self.value * 2

def function2(param1, param2):
    '''Another function'''
    if param1 > param2:
        return param1
    else:
        return param2

# Global variable
CONSTANT = 42
"""

    # Mock parser directly
    if True:
        mock_parser = MagicMock()
        mock_tree = MagicMock()

        # Create mock nodes for different code structures
        nodes = []
        for i, node_type in enumerate(
            [
                "function_definition",
                "class_definition",
                "function_definition",
                "expression_statement",
            ]
        ):
            mock_node = MagicMock()
            mock_node.type = node_type
            mock_node.start_byte = i * 100
            mock_node.end_byte = (i + 1) * 100 - 1
            mock_node.start_point = (i * 5, 0)
            mock_node.end_point = ((i + 1) * 5 - 1, 0)
            mock_node.children = []
            nodes.append(mock_node)

        mock_tree.root_node = MagicMock()
        mock_tree.root_node.children = nodes
        mock_parser.parse.return_value = mock_tree

        chunks = proc._chunk_code_by_ast(python_code.encode(), mock_parser, "python")
        assert len(chunks) > 0
        assert any("function" in chunk.get("metadata", {}).get("type", "") for chunk in chunks)


@pytest.mark.asyncio
async def test_chunk_pdf_content():
    """Test _chunk_pdf_content method (lines 474)."""
    proc = document_processor.DocumentProcessor(
        config.DocumentConfig(), config.ChunkingConfig(max_chunk_size=100)
    )

    # Create pages with varying content sizes
    pages_text = [
        "Short page 1.",
        "Page 2 with more content. " * 10,
        "Page 3 with even more content to test chunking. " * 20,
        "Page 4.",
        "Final page with conclusion. " * 5,
    ]

    chunks = proc._chunk_pdf_content(pages_text)
    assert len(chunks) > 0
    assert any("page" in chunk.get("metadata", {}) for chunk in chunks)

    # Test empty pages
    pages_with_empty = ["Content", "", "More content", "", "Final"]
    chunks = proc._chunk_pdf_content(pages_with_empty)
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_extract_headers_from_html():
    """Test _extract_headers method (lines 516-523)."""
    proc = document_processor.DocumentProcessor(config.DocumentConfig(), config.ChunkingConfig())

    from bs4 import BeautifulSoup

    html = """
    <h1>Main Title</h1>
    <h2>Section 1</h2>
    <h3>Subsection 1.1</h3>
    <h2>Section 2</h2>
    <h4>Deep section</h4>
    <h5>Very deep</h5>
    <h6>Deepest</h6>
    """

    with patch("eol.rag_context.document_processor.BeautifulSoup") as MockBS:
        mock_soup = MagicMock()

        # Create mock header elements
        headers = []
        for i, (tag, text) in enumerate(
            [
                ("h1", "Main Title"),
                ("h2", "Section 1"),
                ("h3", "Subsection 1.1"),
                ("h2", "Section 2"),
                ("h4", "Deep section"),
                ("h5", "Very deep"),
                ("h6", "Deepest"),
            ]
        ):
            mock_header = MagicMock()
            mock_header.name = tag
            mock_header.get_text.return_value = text
            headers.append(mock_header)

        mock_soup.find_all.return_value = headers
        MockBS.return_value = mock_soup

        headers_result = proc._extract_headers(mock_soup)
        assert len(headers_result) > 0
        assert any("Main Title" in h["text"] for h in headers_result)


@pytest.mark.asyncio
async def test_extract_text_content_from_html():
    """Test _extract_text_content method (lines 537-544)."""
    proc = document_processor.DocumentProcessor(config.DocumentConfig(), config.ChunkingConfig())

    with patch("eol.rag_context.document_processor.BeautifulSoup") as MockBS:
        mock_soup = MagicMock()

        # Create mock paragraph elements
        paragraphs = []
        for i in range(5):
            mock_p = MagicMock()
            mock_p.get_text.return_value = f"Paragraph {i+1} content with some text."
            paragraphs.append(mock_p)

        mock_soup.find_all.return_value = paragraphs
        MockBS.return_value = mock_soup

        content_result = proc._extract_text_content(mock_soup)
        assert len(content_result) > 0
        assert any("Paragraph 1" in text for text in content_result)


@pytest.mark.asyncio
async def test_detect_language():
    """Test _detect_language method (lines 554-567)."""
    proc = document_processor.DocumentProcessor(config.DocumentConfig(), config.ChunkingConfig())

    # Test all supported extensions
    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".cs": "csharp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".r": "r",
        ".R": "r",
        ".m": "matlab",
        ".jl": "julia",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".ps1": "powershell",
        ".lua": "lua",
        ".pl": "perl",
        ".unknown": None,
        ".xyz": None,
    }

    for ext, expected_lang in language_map.items():
        lang = proc._detect_language(ext)
        if expected_lang:
            assert lang == expected_lang
        else:
            assert lang is None


@pytest.mark.asyncio
async def test_process_file_with_different_extensions():
    """Test process_file method with various file types."""
    proc = document_processor.DocumentProcessor(config.DocumentConfig(), config.ChunkingConfig())

    # Test code files
    with patch("eol.rag_context.document_processor.aiofiles.open") as mock_aio:
        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value="def test(): pass")
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock()
        mock_aio.return_value = mock_file

        for ext in [".py", ".js", ".java", ".go", ".rs"]:
            doc = await proc.process_file(Path(f"/test{ext}"))
            assert doc.doc_type == "code"

    # Test with magic for unknown extension
    with (
        patch("eol.rag_context.document_processor.magic") as mock_magic,
        patch("eol.rag_context.document_processor.aiofiles.open") as mock_aio,
    ):

        mock_file = AsyncMock()
        mock_file.read = AsyncMock(return_value="content")
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock()
        mock_aio.return_value = mock_file

        # Test PDF detection via magic
        mock_magic.from_file.return_value = "application/pdf"
        with (
            patch("builtins.open", mock_open(read_data=b"PDF")),
            patch("eol.rag_context.document_processor.pypdf.PdfReader") as MockPdf,
        ):
            mock_reader = MagicMock()
            mock_reader.pages = [MagicMock()]
            mock_reader.pages[0].extract_text.return_value = "PDF content"
            MockPdf.return_value = mock_reader

            doc = await proc.process_file(Path("/test.xyz"))
            assert doc is not None

        # Test DOCX detection via magic
        mock_magic.from_file.return_value = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        with patch("eol.rag_context.document_processor.docx.Document") as MockDocx:
            mock_doc = MagicMock()
            mock_doc.paragraphs = [MagicMock(text="Word content")]
            mock_doc.tables = []
            MockDocx.return_value = mock_doc

            doc = await proc.process_file(Path("/test.abc"))
            assert doc is not None


@pytest.mark.asyncio
async def test_process_structured_files():
    """Test _process_structured method for JSON and YAML."""
    proc = document_processor.DocumentProcessor(config.DocumentConfig(), config.ChunkingConfig())

    # Test JSON file
    json_data = {
        "name": "test",
        "config": {"setting1": "value1", "setting2": 123, "nested": {"deep": "value"}},
        "items": ["item1", "item2", "item3"],
    }

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(json_data, f)
        f.flush()

        doc = await proc._process_structured(Path(f.name))
        assert doc.doc_type == "structured"
        assert "test" in doc.content
        assert len(doc.chunks) > 0

        import os

        os.unlink(f.name)

    # Test YAML file
    yaml_content = """
name: test
config:
  setting1: value1
  setting2: 123
  nested:
    deep: value
items:
  - item1
  - item2
  - item3
"""

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(yaml_content)
        f.flush()

        with patch("eol.rag_context.document_processor.yaml") as mock_yaml:
            mock_yaml.safe_load.return_value = {
                "name": "test",
                "config": {"setting1": "value1", "setting2": 123},
                "items": ["item1", "item2", "item3"],
            }

            doc = await proc._process_structured(Path(f.name))
            assert doc.doc_type == "structured"
            assert len(doc.chunks) > 0

        import os

        os.unlink(f.name)


@pytest.mark.asyncio
async def test_chunk_structured_data():
    """Test _chunk_structured_data method."""
    proc = document_processor.DocumentProcessor(config.DocumentConfig(), config.ChunkingConfig())

    # Test with nested dictionary
    nested_data = {
        "level1": {
            "level2": {"level3": {"data": "deep value"}},
            "array": [1, 2, 3, 4, 5],
            "text": "Some text value",
        },
        "another_key": "another value",
    }

    chunks = proc._chunk_structured_data(nested_data, "json")
    assert len(chunks) > 0
    assert any("level1" in chunk["content"] for chunk in chunks)

    # Test with array
    array_data = [
        {"id": 1, "name": "Item 1"},
        {"id": 2, "name": "Item 2"},
        {"id": 3, "name": "Item 3"},
    ]

    chunks = proc._chunk_structured_data(array_data, "json")
    assert len(chunks) > 0

    # Test with large data
    large_data = {f"key_{i}": f"value_{i}" * 10 for i in range(100)}

    chunks = proc._chunk_structured_data(large_data, "json")
    assert len(chunks) > 1  # Should be chunked due to size
