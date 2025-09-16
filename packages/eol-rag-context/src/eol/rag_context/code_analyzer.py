"""
Code analysis module for extracting entities and relationships from source code.
Uses Python's built-in AST module for Python code analysis.
"""

import ast
import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CodeEntityType(Enum):
    """Types of entities that can be extracted from code."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    DECORATOR = "decorator"
    PARAMETER = "parameter"
    ATTRIBUTE = "attribute"
    CONSTANT = "constant"


class CodeRelationType(Enum):
    """Types of relationships between code entities."""

    IMPORTS = "imports"
    DEFINES = "defines"
    CALLS = "calls"
    INHERITS_FROM = "inherits_from"
    DECORATES = "decorates"
    USES = "uses"
    RETURNS = "returns"
    RAISES = "raises"
    OVERRIDES = "overrides"
    IMPLEMENTS = "implements"
    INSTANTIATES = "instantiates"
    ASSIGNS_TO = "assigns_to"
    REFERENCES = "references"


@dataclass
class CodeEntity:
    """Represents a code entity extracted from source."""

    id: str
    type: CodeEntityType
    name: str
    file_path: str
    line_start: int
    line_end: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    docstring: Optional[str] = None
    signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content": self.content[:500],  # Truncate for storage
            "metadata": self.metadata,
            "docstring": self.docstring[:500] if self.docstring else None,
            "signature": self.signature,
        }


@dataclass
class CodeRelation:
    """Represents a relationship between code entities."""

    source_id: str
    target_id: str
    type: CodeRelationType
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to dictionary representation."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type.value,
            "metadata": self.metadata,
        }


class ASTCodeAnalyzer:
    """Analyzes Python code using AST to extract entities and relationships."""

    def __init__(self):
        """Initialize the code analyzer."""
        self.entities: List[CodeEntity] = []
        self.relations: List[CodeRelation] = []
        self._entity_map: Dict[str, CodeEntity] = {}
        self._current_file: Optional[str] = None
        self._current_class: Optional[str] = None
        self._import_map: Dict[str, str] = {}

    def analyze_file(self, file_path: Path) -> Tuple[List[CodeEntity], List[CodeRelation]]:
        """
        Analyze a Python file to extract entities and relationships.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            Tuple of (entities, relations) extracted from the file
        """
        self.entities = []
        self.relations = []
        self._entity_map = {}
        self._current_file = str(file_path)
        self._import_map = {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))
            self._visit_node(tree, content)

            return self.entities, self.relations

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return [], []

    def _generate_entity_id(self, entity_type: str, name: str, context: str = "") -> str:
        """Generate a unique ID for an entity."""
        components = [self._current_file, entity_type, name]
        if context:
            components.append(context)
        id_string = ":".join(components)
        return hashlib.md5(id_string.encode()).hexdigest()[:16]

    def _visit_node(self, node: ast.AST, source: str, parent_id: Optional[str] = None) -> None:
        """Recursively visit AST nodes to extract entities and relationships."""
        if isinstance(node, ast.Module):
            self._process_module(node, source, parent_id)
            # Module processing handles its own children
            return
        elif isinstance(node, ast.ClassDef):
            self._process_class(node, source, parent_id)
            # Class processing handles its own children
            return
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            self._process_function(node, source, parent_id)
            # Function processing handles its own children via _extract_function_calls
            return
        elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            self._process_import(node, source, parent_id)
        elif isinstance(node, ast.Assign):
            self._process_assignment(node, source, parent_id)

        # Recursively process child nodes (only for nodes not handled above)
        for child in ast.iter_child_nodes(node):
            self._visit_node(child, source, parent_id)

    def _process_module(
        self, node: ast.Module, source: str, parent_id: Optional[str] = None
    ) -> None:
        """Process a module node."""
        module_name = Path(self._current_file).stem
        entity_id = self._generate_entity_id("module", module_name)

        # Extract module docstring
        docstring = ast.get_docstring(node)

        entity = CodeEntity(
            id=entity_id,
            type=CodeEntityType.MODULE,
            name=module_name,
            file_path=self._current_file,
            line_start=1,
            line_end=len(source.splitlines()),
            content=source[:1000],  # Store first 1000 chars
            docstring=docstring,
            metadata={"node_count": len(list(ast.walk(node)))},
        )

        self.entities.append(entity)
        self._entity_map[entity_id] = entity

        # Process module body
        for item in node.body:
            self._visit_node(item, source, entity_id)

    def _process_class(
        self, node: ast.ClassDef, source: str, parent_id: Optional[str] = None
    ) -> None:
        """Process a class definition."""
        entity_id = self._generate_entity_id("class", node.name)
        self._current_class = node.name

        # Extract class signature
        bases = [ast.unparse(base) for base in node.bases]
        signature = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Extract source lines
        lines = source.splitlines()
        content = "\n".join(lines[node.lineno - 1 : node.end_lineno])

        entity = CodeEntity(
            id=entity_id,
            type=CodeEntityType.CLASS,
            name=node.name,
            file_path=self._current_file,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            content=content[:1000],
            signature=signature,
            docstring=docstring,
            metadata={
                "bases": bases,
                "decorators": [ast.unparse(d) for d in node.decorator_list],
                "methods": [],
            },
        )

        self.entities.append(entity)
        self._entity_map[entity_id] = entity

        # Add relationship to parent
        if parent_id:
            self.relations.append(
                CodeRelation(
                    source_id=parent_id, target_id=entity_id, type=CodeRelationType.DEFINES
                )
            )

        # Process inheritance relationships
        for base in node.bases:
            base_name = ast.unparse(base)
            base_id = self._generate_entity_id("class", base_name)
            self.relations.append(
                CodeRelation(
                    source_id=entity_id, target_id=base_id, type=CodeRelationType.INHERITS_FROM
                )
            )

        # Process class body
        for item in node.body:
            self._visit_node(item, source, entity_id)

        self._current_class = None

    def _process_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source: str,
        parent_id: Optional[str] = None,
    ) -> None:
        """Process a function or method definition."""
        is_method = self._current_class is not None
        entity_type = CodeEntityType.METHOD if is_method else CodeEntityType.FUNCTION
        context = self._current_class if is_method else ""
        entity_id = self._generate_entity_id(entity_type.value, node.name, context)

        # Extract function signature
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        signature = f"{async_prefix}def {node.name}({', '.join(args)}){returns}"

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Extract source lines
        lines = source.splitlines()
        content = "\n".join(lines[node.lineno - 1 : node.end_lineno])

        entity = CodeEntity(
            id=entity_id,
            type=entity_type,
            name=node.name,
            file_path=self._current_file,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            content=content[:1000],
            signature=signature,
            docstring=docstring,
            metadata={
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "decorators": [ast.unparse(d) for d in node.decorator_list],
                "parameters": [arg.arg for arg in node.args.args],
            },
        )

        self.entities.append(entity)
        self._entity_map[entity_id] = entity

        # Add relationship to parent
        if parent_id:
            self.relations.append(
                CodeRelation(
                    source_id=parent_id, target_id=entity_id, type=CodeRelationType.DEFINES
                )
            )

        # Update parent's metadata if it's a class
        if is_method and parent_id in self._entity_map:
            parent = self._entity_map[parent_id]
            if "methods" in parent.metadata:
                parent.metadata["methods"].append(node.name)

        # Process function body for calls
        self._extract_function_calls(node, entity_id)

    def _process_import(
        self, node: ast.Import | ast.ImportFrom, source: str, parent_id: Optional[str] = None
    ) -> None:
        """Process import statements."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                as_name = alias.asname or alias.name
                self._import_map[as_name] = module_name

                entity_id = self._generate_entity_id("import", module_name)
                entity = CodeEntity(
                    id=entity_id,
                    type=CodeEntityType.IMPORT,
                    name=module_name,
                    file_path=self._current_file,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    content=f"import {module_name}",
                    metadata={"as_name": as_name if alias.asname else None},
                )
                self.entities.append(entity)

                if parent_id:
                    self.relations.append(
                        CodeRelation(
                            source_id=parent_id, target_id=entity_id, type=CodeRelationType.IMPORTS
                        )
                    )

        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            for alias in node.names:
                import_name = alias.name
                as_name = alias.asname or alias.name
                full_name = f"{module_name}.{import_name}" if module_name else import_name
                self._import_map[as_name] = full_name

                entity_id = self._generate_entity_id("import", full_name)
                entity = CodeEntity(
                    id=entity_id,
                    type=CodeEntityType.IMPORT,
                    name=full_name,
                    file_path=self._current_file,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    content=f"from {module_name} import {import_name}",
                    metadata={
                        "module": module_name,
                        "name": import_name,
                        "as_name": as_name if alias.asname else None,
                    },
                )
                self.entities.append(entity)

                if parent_id:
                    self.relations.append(
                        CodeRelation(
                            source_id=parent_id, target_id=entity_id, type=CodeRelationType.IMPORTS
                        )
                    )

    def _process_assignment(
        self, node: ast.Assign, source: str, parent_id: Optional[str] = None
    ) -> None:
        """Process variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Determine if it's a constant (uppercase) or variable
                entity_type = (
                    CodeEntityType.CONSTANT if target.id.isupper() else CodeEntityType.VARIABLE
                )

                entity_id = self._generate_entity_id(
                    entity_type.value, target.id, self._current_class or ""
                )

                # Extract the assignment line
                lines = source.splitlines()
                content = lines[node.lineno - 1] if node.lineno <= len(lines) else ""

                entity = CodeEntity(
                    id=entity_id,
                    type=entity_type,
                    name=target.id,
                    file_path=self._current_file,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    content=content,
                    metadata={"value_type": type(node.value).__name__},
                )
                self.entities.append(entity)

                if parent_id:
                    self.relations.append(
                        CodeRelation(
                            source_id=parent_id, target_id=entity_id, type=CodeRelationType.DEFINES
                        )
                    )

    def _extract_function_calls(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, caller_id: str
    ) -> None:
        """Extract function calls from a function body."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                # Try to extract the function being called
                if isinstance(child.func, ast.Name):
                    called_name = child.func.id
                    called_id = self._generate_entity_id("function", called_name)
                    self.relations.append(
                        CodeRelation(
                            source_id=caller_id, target_id=called_id, type=CodeRelationType.CALLS
                        )
                    )
                elif isinstance(child.func, ast.Attribute):
                    # Method call like obj.method()
                    method_name = child.func.attr
                    if isinstance(child.func.value, ast.Name):
                        obj_name = child.func.value.id
                        called_id = self._generate_entity_id("method", method_name, obj_name)
                        self.relations.append(
                            CodeRelation(
                                source_id=caller_id,
                                target_id=called_id,
                                type=CodeRelationType.CALLS,
                                metadata={"object": obj_name},
                            )
                        )

    def analyze_directory(self, directory: Path) -> Tuple[List[CodeEntity], List[CodeRelation]]:
        """
        Analyze all Python files in a directory.

        Args:
            directory: Path to the directory to analyze

        Returns:
            Tuple of (all_entities, all_relations) from all files
        """
        all_entities = []
        all_relations = []

        for file_path in directory.rglob("*.py"):
            entities, relations = self.analyze_file(file_path)
            all_entities.extend(entities)
            all_relations.extend(relations)

        return all_entities, all_relations
