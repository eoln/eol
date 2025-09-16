"""
Unit tests for the code analyzer module.
"""

import pytest

from eol.rag_context.code_analyzer import (
    ASTCodeAnalyzer,
    CodeEntity,
    CodeEntityType,
    CodeRelation,
    CodeRelationType,
)


class TestCodeAnalyzer:
    """Test the ASTCodeAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create an analyzer instance."""
        return ASTCodeAnalyzer()

    @pytest.fixture
    def sample_python_code(self):
        """Sample Python code for testing."""
        return '''
"""Module docstring."""
import os
from typing import List

class MyClass:
    """Class docstring."""

    def __init__(self, name: str):
        """Initialize the class."""
        self.name = name

    def process(self, data: List[str]) -> str:
        """Process data."""
        result = self.helper(data)
        return result

    def helper(self, items: List[str]) -> str:
        """Helper method."""
        return ", ".join(items)

def standalone_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

CONSTANT = 42
variable = "test"
'''

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert analyzer.entities == []
        assert analyzer.relations == []
        assert analyzer._entity_map == {}

    def test_analyze_file_with_class(self, analyzer, sample_python_code, tmp_path):
        """Test analyzing a file with a class definition."""
        # Create temporary file
        test_file = tmp_path / "test.py"
        test_file.write_text(sample_python_code)

        # Analyze the file
        entities, relations = analyzer.analyze_file(test_file)

        # Check entities were extracted
        assert len(entities) > 0

        # Check for specific entity types
        entity_types = {e.type for e in entities}
        assert CodeEntityType.MODULE in entity_types
        assert CodeEntityType.CLASS in entity_types
        assert CodeEntityType.METHOD in entity_types
        assert CodeEntityType.FUNCTION in entity_types
        assert CodeEntityType.IMPORT in entity_types

        # Check for specific entities
        entity_names = {e.name for e in entities}
        assert "MyClass" in entity_names
        assert "__init__" in entity_names
        assert "process" in entity_names
        assert "helper" in entity_names
        assert "standalone_function" in entity_names

        # Check relations were extracted
        assert len(relations) > 0
        relation_types = {r.type for r in relations}
        assert CodeRelationType.DEFINES in relation_types

    def test_analyze_simple_function(self, analyzer, tmp_path):
        """Test analyzing a simple function."""
        code = """
def simple_function():
    \"\"\"Simple function docstring.\"\"\"
    return 42
"""
        test_file = tmp_path / "simple.py"
        test_file.write_text(code)

        entities, relations = analyzer.analyze_file(test_file)

        # Find the function entity
        func_entities = [e for e in entities if e.type == CodeEntityType.FUNCTION]
        assert len(func_entities) == 1
        func = func_entities[0]
        assert func.name == "simple_function"
        assert func.docstring == "Simple function docstring."
        assert "def simple_function()" in func.signature

    def test_analyze_class_inheritance(self, analyzer, tmp_path):
        """Test analyzing class inheritance."""
        code = """
class BaseClass:
    pass

class DerivedClass(BaseClass):
    pass
"""
        test_file = tmp_path / "inheritance.py"
        test_file.write_text(code)

        entities, relations = analyzer.analyze_file(test_file)

        # Check inheritance relationship
        inheritance_relations = [r for r in relations if r.type == CodeRelationType.INHERITS_FROM]
        assert len(inheritance_relations) > 0

    def test_analyze_imports(self, analyzer, tmp_path):
        """Test analyzing import statements."""
        code = """
import os
import sys as system
from typing import List, Dict
from pathlib import Path
"""
        test_file = tmp_path / "imports.py"
        test_file.write_text(code)

        entities, relations = analyzer.analyze_file(test_file)

        # Check import entities
        import_entities = [e for e in entities if e.type == CodeEntityType.IMPORT]
        assert len(import_entities) == 5  # os, sys, List, Dict, Path

        import_names = {e.name for e in import_entities}
        assert "os" in import_names
        assert "sys" in import_names
        assert "typing.List" in import_names
        assert "typing.Dict" in import_names
        assert "pathlib.Path" in import_names

    def test_analyze_async_function(self, analyzer, tmp_path):
        """Test analyzing async functions."""
        code = """
async def async_function(data: str) -> str:
    \"\"\"Async function.\"\"\"
    return data.upper()
"""
        test_file = tmp_path / "async.py"
        test_file.write_text(code)

        entities, relations = analyzer.analyze_file(test_file)

        # Find the async function
        func_entities = [e for e in entities if e.type == CodeEntityType.FUNCTION]
        assert len(func_entities) == 1
        func = func_entities[0]
        assert func.name == "async_function"
        assert "async def" in func.signature
        assert func.metadata["is_async"] is True

    def test_analyze_constants_and_variables(self, analyzer, tmp_path):
        """Test analyzing constants and variables."""
        code = """
CONSTANT_VALUE = 100
another_variable = "test"
"""
        test_file = tmp_path / "variables.py"
        test_file.write_text(code)

        entities, relations = analyzer.analyze_file(test_file)

        # Check for constant and variable entities
        constant_entities = [e for e in entities if e.type == CodeEntityType.CONSTANT]
        variable_entities = [e for e in entities if e.type == CodeEntityType.VARIABLE]

        assert len(constant_entities) == 1
        assert constant_entities[0].name == "CONSTANT_VALUE"

        assert len(variable_entities) == 1
        assert variable_entities[0].name == "another_variable"

    def test_analyze_method_calls(self, analyzer, tmp_path):
        """Test extraction of method calls."""
        code = """
class MyClass:
    def method_a(self):
        self.method_b()
        helper_function()

    def method_b(self):
        pass

def helper_function():
    pass
"""
        test_file = tmp_path / "calls.py"
        test_file.write_text(code)

        entities, relations = analyzer.analyze_file(test_file)

        # Check for call relationships
        call_relations = [r for r in relations if r.type == CodeRelationType.CALLS]
        assert len(call_relations) > 0

    def test_analyze_decorators(self, analyzer, tmp_path):
        """Test analyzing decorated functions."""
        code = """
@property
def my_property(self):
    return self._value

@staticmethod
def static_method():
    pass

@classmethod
def class_method(cls):
    pass
"""
        test_file = tmp_path / "decorators.py"
        test_file.write_text(code)

        entities, relations = analyzer.analyze_file(test_file)

        # Check that decorators are captured in metadata
        func_entities = [e for e in entities if e.type == CodeEntityType.FUNCTION]
        for func in func_entities:
            if func.name == "my_property":
                assert "property" in str(func.metadata.get("decorators", []))

    def test_analyze_directory(self, analyzer, tmp_path):
        """Test analyzing multiple files in a directory."""
        # Create multiple Python files
        (tmp_path / "file1.py").write_text("def func1(): pass")
        (tmp_path / "file2.py").write_text("class Class2: pass")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.py").write_text("CONSTANT = 42")

        # Analyze directory
        entities, relations = analyzer.analyze_directory(tmp_path)

        # Should have entities from all files
        assert len(entities) > 0
        entity_names = {e.name for e in entities}
        assert "func1" in entity_names
        assert "Class2" in entity_names
        assert "CONSTANT" in entity_names

    def test_analyze_malformed_code(self, analyzer, tmp_path):
        """Test handling of malformed code."""
        code = "def broken_function(:\n    pass"  # Syntax error
        test_file = tmp_path / "broken.py"
        test_file.write_text(code)

        # Should handle error gracefully
        entities, relations = analyzer.analyze_file(test_file)
        assert entities == []
        assert relations == []

    def test_entity_to_dict(self):
        """Test CodeEntity.to_dict method."""
        entity = CodeEntity(
            id="test_id",
            type=CodeEntityType.FUNCTION,
            name="test_func",
            file_path="/path/to/file.py",
            line_start=1,
            line_end=5,
            content="def test_func(): pass",
            metadata={"test": "value"},
            docstring="Test docstring",
            signature="def test_func()",
        )

        result = entity.to_dict()
        assert result["id"] == "test_id"
        assert result["type"] == "function"
        assert result["name"] == "test_func"
        assert result["docstring"] == "Test docstring"

    def test_relation_to_dict(self):
        """Test CodeRelation.to_dict method."""
        relation = CodeRelation(
            source_id="source",
            target_id="target",
            type=CodeRelationType.CALLS,
            metadata={"test": "value"},
        )

        result = relation.to_dict()
        assert result["source_id"] == "source"
        assert result["target_id"] == "target"
        assert result["type"] == "calls"
        assert result["metadata"]["test"] == "value"

    def test_generate_entity_id(self, analyzer):
        """Test entity ID generation."""
        analyzer._current_file = "/path/to/file.py"
        id1 = analyzer._generate_entity_id("function", "my_func")
        id2 = analyzer._generate_entity_id("function", "my_func")
        id3 = analyzer._generate_entity_id("function", "other_func")

        # Same inputs should generate same ID
        assert id1 == id2
        # Different inputs should generate different IDs
        assert id1 != id3
        # IDs should be 16 characters (truncated MD5)
        assert len(id1) == 16
