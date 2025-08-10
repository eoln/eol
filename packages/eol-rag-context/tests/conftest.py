"""
Simplified pytest configuration for unit tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict
import numpy as np


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_documents(temp_dir: Path) -> Dict[str, Path]:
    """Create sample documents for testing."""
    docs = {}
    
    # Markdown file
    md_file = temp_dir / "test.md"
    md_file.write_text("""# Test Document

## Introduction
This is a test document for RAG indexing.

## Code Example
```python
def hello_world():
    print("Hello, World!")
```

## Conclusion
This concludes our test document.
""")
    docs["markdown"] = md_file
    
    # Python file
    py_file = temp_dir / "test.py"
    py_file.write_text("""
def factorial(n):
    \"\"\"Calculate factorial of n.\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n - 1)

class Calculator:
    \"\"\"Simple calculator class.\"\"\"
    
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b
""")
    docs["python"] = py_file
    
    # JSON file
    json_file = temp_dir / "config.json"
    json_file.write_text("""{
    "name": "test-project",
    "version": "1.0.0",
    "dependencies": {
        "redis": "^5.0.0",
        "numpy": "^1.24.0"
    }
}""")
    docs["json"] = json_file
    
    # Text file
    txt_file = temp_dir / "readme.txt"
    txt_file.write_text("""This is a sample project for testing the RAG system.

It includes various file types to test the document processor.
The system should handle markdown, code, JSON, and plain text files.

Features:
- Document indexing
- Vector search
- Knowledge graph
- Real-time updates
""")
    docs["text"] = txt_file
    
    return docs