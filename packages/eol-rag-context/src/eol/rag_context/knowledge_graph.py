"""Knowledge graph builder for semantic knowledge discovery and relationship mapping.

This module constructs and manages dynamic knowledge graphs from indexed documents,
enabling advanced semantic search and knowledge discovery capabilities. The knowledge
graph captures entities (concepts, functions, classes, topics) and their relationships
(structural, semantic, temporal) to provide rich contextual information for RAG systems.

Key Features:
    - Multi-level entity extraction (documents, concepts, code elements, topics)
    - Semantic relationship discovery using vector similarity
    - Code structure analysis with function/class relationships
    - Graph-based query processing for contextual information retrieval
    - Community detection and pattern discovery for knowledge insights
    - NetworkX integration for advanced graph algorithms
    - Redis-backed persistent storage with efficient querying

The knowledge graph operates on multiple entity types (documents, concepts, functions,
classes, topics, terms) and relationship types (structural, semantic, code-specific)
to create a comprehensive representation of the codebase and documentation.

Example:
    Building and querying a knowledge graph:

    >>> from eol.rag_context.knowledge_graph import KnowledgeGraphBuilder
    >>>
    >>> # Initialize builder
    >>> kg_builder = KnowledgeGraphBuilder(redis_store, embedding_manager)
    >>>
    >>> # Build graph from indexed documents
    >>> await kg_builder.build_from_documents(source_id="project_docs")
    >>>
    >>> # Query for relevant subgraph
    >>> subgraph = await kg_builder.query_subgraph(
    ...     query="authentication and security",
    ...     max_depth=2,
    ...     max_entities=15
    ... )
    >>>
    >>> # Analyze results
    >>> print(f"Found {len(subgraph.entities)} related entities")
    >>> for entity in subgraph.entities[:5]:
    ...     print(f"- {entity.name} ({entity.type.value})")
    >>>
    >>> # Discover patterns
    >>> patterns = await kg_builder.discover_patterns(min_support=0.1)
    >>> for pattern in patterns[:3]:
    ...     print(f"Pattern: {pattern['pattern']} (support: {pattern['support']:.1%})")
"""

import asyncio
import json
import hashlib
from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np
import networkx as nx
import logging
from enum import Enum

from .redis_client import RedisVectorStore
from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Enumeration of entity types supported in the knowledge graph.

    Defines the various types of entities that can be extracted and stored
    in the knowledge graph. Each entity type represents a different kind of
    semantic unit with specific properties and relationship patterns.

    Entity Categories:
        Structural: DOCUMENT, SECTION, MODULE, FILE
        Code Elements: FUNCTION, CLASS, API
        Conceptual: CONCEPT, TOPIC, TERM
        External: PERSON, ORGANIZATION, TECHNOLOGY, DATABASE, URL

    Example:
        >>> entity_type = EntityType.FUNCTION
        >>> print(entity_type.value)  # "function"
        >>> print(EntityType.CONCEPT in [EntityType.CONCEPT, EntityType.TOPIC])
        True
    """

    CONCEPT = "concept"
    DOCUMENT = "document"
    SECTION = "section"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    TOPIC = "topic"
    TERM = "term"
    PERSON = "person"
    ORGANIZATION = "organization"
    TECHNOLOGY = "technology"
    API = "api"
    DATABASE = "database"
    FILE = "file"
    URL = "url"


class RelationType(Enum):
    """Enumeration of relationship types for connecting knowledge graph entities.

    Defines the various relationship types that can exist between entities in
    the knowledge graph. Relationships capture both structural organization
    and semantic connections to enable rich graph traversal and discovery.

    Relationship Categories:
        Structural: CONTAINS, PART_OF, PARENT_OF, CHILD_OF
        Semantic: RELATES_TO, SIMILAR_TO, DEPENDS_ON, IMPLEMENTS, EXTENDS
        Code-Specific: USES, CALLS, IMPORTS
        Documentation: DESCRIBES, REFERENCES, DEFINES, EXPLAINS
        Temporal: BEFORE, AFTER, DURING
        Comparative: SAME_AS, DIFFERENT_FROM, ALTERNATIVE_TO

    Example:
        >>> rel_type = RelationType.SIMILAR_TO
        >>> print(rel_type.value)  # "similar_to"
        >>>
        >>> # Check if relationship is structural
        >>> structural_types = {RelationType.CONTAINS, RelationType.PART_OF}
        >>> is_structural = rel_type in structural_types
    """

    # Structural relationships
    CONTAINS = "contains"
    PART_OF = "part_of"
    PARENT_OF = "parent_of"
    CHILD_OF = "child_of"

    # Semantic relationships
    RELATES_TO = "relates_to"
    SIMILAR_TO = "similar_to"
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    EXTENDS = "extends"
    USES = "uses"
    CALLS = "calls"
    IMPORTS = "imports"

    # Documentation relationships
    DESCRIBES = "describes"
    REFERENCES = "references"
    DEFINES = "defines"
    EXPLAINS = "explains"

    # Temporal relationships
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"

    # Comparison relationships
    SAME_AS = "same_as"
    DIFFERENT_FROM = "different_from"
    ALTERNATIVE_TO = "alternative_to"


@dataclass
class Entity:
    """Knowledge graph entity representing a semantic unit with properties and relationships.

    Represents a single entity in the knowledge graph, containing all necessary
    information for semantic processing, similarity matching, and relationship
    building. Entities can represent anything from code functions to abstract concepts.

    Attributes:
        id: Unique identifier for the entity (format: "{type}_{hash}" or custom).
        name: Human-readable name or title of the entity.
        type: EntityType enum value specifying the kind of entity.
        content: Full or partial text content associated with the entity.
        embedding: Optional vector embedding for similarity calculations.
        properties: Additional metadata as key-value pairs.
        source_ids: Set of source document IDs where this entity appears.

    Example:
        Creating a function entity:

        >>> import numpy as np
        >>> from eol.rag_context.knowledge_graph import Entity, EntityType
        >>>
        >>> entity = Entity(
        ...     id="func_auth_validate_token",
        ...     name="validate_token",
        ...     type=EntityType.FUNCTION,
        ...     content="Function to validate JWT tokens",
        ...     embedding=np.random.rand(384).astype(np.float32),
        ...     properties={
        ...         "language": "python",
        ...         "parameters": ["token", "secret"],
        ...         "returns": "bool"
        ...     },
        ...     source_ids={"auth_module_v1"}
        ... )
        >>> print(f"{entity.name} ({entity.type.value})")
        validate_token (function)
    """

    id: str
    name: str
    type: EntityType
    content: str = ""
    embedding: Optional[np.ndarray] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    source_ids: Set[str] = field(default_factory=set)


@dataclass
class Relationship:
    """Directed relationship between two entities in the knowledge graph.

    Represents a typed, weighted connection between two entities, enabling
    graph traversal, relationship analysis, and contextual information retrieval.
    Relationships can have additional properties for storing relationship-specific data.

    Attributes:
        source_id: ID of the source entity (relationship origin).
        target_id: ID of the target entity (relationship destination).
        type: RelationType enum value specifying the relationship kind.
        weight: Numeric weight indicating relationship strength (default: 1.0).
        properties: Additional relationship metadata as key-value pairs.

    Example:
        Creating a similarity relationship:

        >>> from eol.rag_context.knowledge_graph import Relationship, RelationType
        >>>
        >>> relationship = Relationship(
        ...     source_id="func_auth_login",
        ...     target_id="func_auth_validate",
        ...     type=RelationType.SIMILAR_TO,
        ...     weight=0.87,
        ...     properties={
        ...         "similarity": 0.87,
        ...         "common_tokens": ["auth", "token", "validate"]
        ...     }
        ... )
        >>> print(f"Relationship: {relationship.type.value} (weight: {relationship.weight})")
        Relationship: similar_to (weight: 0.87)
    """

    source_id: str
    target_id: str
    type: RelationType
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeSubgraph:
    """Query-specific subgraph containing relevant entities and relationships.

    Represents a subset of the knowledge graph that is relevant to a specific
    query or context. Used for providing focused knowledge context to LLMs
    while maintaining computational efficiency.

    Attributes:
        entities: List of entities included in the subgraph.
        relationships: List of relationships between included entities.
        central_entities: List of entity IDs that were most relevant to the query.
        metadata: Additional subgraph information including query context.

    Example:
        Processing a query subgraph:

        >>> subgraph = await kg_builder.query_subgraph("database connections")
        >>> print(f"Found {len(subgraph.entities)} entities, {len(subgraph.relationships)} relationships")
        >>>
        >>> # Focus on central entities first
        >>> central = [e for e in subgraph.entities if e.id in subgraph.central_entities]
        >>> for entity in central:
        ...     print(f"Central: {entity.name} ({entity.type.value})")
        >>>
        >>> # Analyze relationship patterns
        >>> rel_types = {r.type.value for r in subgraph.relationships}
        >>> print(f"Relationship types: {rel_types}")
    """

    entities: List[Entity]
    relationships: List[Relationship]
    central_entities: List[str]  # Most relevant entity IDs
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraphBuilder:
    """Comprehensive knowledge graph builder and manager for semantic information extraction.

    Constructs, maintains, and queries a dynamic knowledge graph from indexed documents,
    extracting entities and relationships to enable advanced semantic search and
    knowledge discovery. Integrates with Redis for persistent storage and NetworkX
    for graph algorithms.

    The builder processes multiple document types (code files, markdown, plain text)
    and extracts various entity types (functions, classes, concepts, topics) along
    with their relationships (structural, semantic, code-specific). This creates a
    rich knowledge representation for contextual information retrieval.

    Key Capabilities:
        - Multi-format document processing (code, markdown, text)
        - Entity extraction with type classification and embedding generation
        - Relationship discovery through structural analysis and semantic similarity
        - Graph-based querying with depth-limited traversal
        - Pattern discovery and community detection
        - Persistent storage in Redis with efficient retrieval

    Attributes:
        redis: Redis vector store for persistent graph storage.
        embeddings: Embedding manager for vector similarity calculations.
        graph: NetworkX MultiDiGraph for in-memory graph operations.
        entities: Dictionary mapping entity IDs to Entity objects.
        relationships: List of all relationships in the graph.

    Example:
        Complete knowledge graph workflow:

        >>> from eol.rag_context.knowledge_graph import KnowledgeGraphBuilder
        >>>
        >>> # Initialize with dependencies
        >>> kg_builder = KnowledgeGraphBuilder(redis_store, embedding_manager)
        >>>
        >>> # Build graph from specific source
        >>> await kg_builder.build_from_documents(
        ...     source_id="my_project",
        ...     max_documents=1000
        ... )
        >>>
        >>> # Get comprehensive statistics
        >>> stats = kg_builder.get_graph_stats()
        >>> print(f"Graph: {stats['entity_count']} entities, {stats['relationship_count']} relationships")
        >>> print(f"Density: {stats['density']:.3f}, Components: {stats['connected_components']}")
        >>>
        >>> # Query for specific information
        >>> subgraph = await kg_builder.query_subgraph(
        ...     query="machine learning algorithms",
        ...     max_depth=3,
        ...     max_entities=25
        ... )
        >>>
        >>> # Discover structural patterns
        >>> patterns = await kg_builder.discover_patterns(min_support=0.05)
        >>> for pattern in patterns:
        ...     if pattern['pattern'].startswith('function'):
        ...         print(f"Code pattern: {pattern['pattern']} ({pattern['support']:.1%} support)")
    """

    def __init__(self, redis_store: RedisVectorStore, embedding_manager: EmbeddingManager):
        """Initialize knowledge graph builder with required dependencies.

        Args:
            redis_store: Redis vector store for graph persistence and vector operations.
            embedding_manager: Manager for generating and caching entity embeddings.
        """
        self.redis = redis_store
        self.embeddings = embedding_manager
        self.graph = nx.MultiDiGraph()  # Multi-directed graph for multiple relationship types
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []

    async def build_from_documents(
        self, source_id: Optional[str] = None, max_documents: Optional[int] = None
    ) -> None:
        """Build comprehensive knowledge graph from indexed documents.

        Constructs a complete knowledge graph by extracting entities and relationships
        from indexed documents across all hierarchy levels (concepts, sections, chunks).
        The process includes entity extraction, relationship discovery, semantic
        similarity analysis, and persistent storage in Redis.

        Build Process:
        1. Extract document-level entities from chunks with metadata
        2. Extract code-specific entities (functions, classes, imports)
        3. Extract conceptual entities from high-level documents
        4. Build structural relationships (parent-child, contains)
        5. Build semantic relationships using vector similarity
        6. Build code-specific relationships (calls, inheritance)
        7. Store complete graph in Redis for persistence

        Args:
            source_id: Optional source ID to limit graph building to specific documents.
                If None, processes all indexed documents.
            max_documents: Optional limit on number of documents to process.
                Useful for large corpora or testing scenarios.

        Example:
            Build graph for entire project:

            >>> await kg_builder.build_from_documents()
            >>> stats = kg_builder.get_graph_stats()
            >>> print(f"Built graph: {stats['entity_count']} entities")

            Build graph for specific source:

            >>> await kg_builder.build_from_documents(
            ...     source_id="auth_module",
            ...     max_documents=50
            ... )

        Note:
            This operation can be time-intensive for large document sets.
            Progress is logged at INFO level for monitoring.
        """
        logger.info("Building knowledge graph from documents...")

        # Extract entities from different hierarchy levels
        await self._extract_document_entities(source_id, max_documents)
        await self._extract_code_entities(source_id)
        await self._extract_conceptual_entities(source_id)

        # Build relationships
        await self._build_structural_relationships()
        await self._build_semantic_relationships()
        await self._build_code_relationships()

        # Store graph in Redis
        await self._store_graph()

        logger.info(
            f"Built knowledge graph with {len(self.entities)} entities and {len(self.relationships)} relationships"
        )

    async def _extract_document_entities(
        self, source_id: Optional[str] = None, max_documents: Optional[int] = None
    ) -> None:
        """Extract entities from document chunks with content analysis.

        Scans Redis for indexed document chunks and creates entity objects
        for each document. Analyzes document content to extract nested
        entities based on document type (code, markdown, text).

        Args:
            source_id: Optional filter for specific source documents.
            max_documents: Optional limit on documents to process.

        Note:
            This is the primary entity extraction method that processes
            chunk-level documents and delegates to content-specific extractors.
        """
        # Scan for documents
        pattern = f"chunk:{source_id}*" if source_id else "chunk:*"
        cursor = 0
        doc_count = 0

        while True:
            cursor, keys = self.redis.redis.scan(cursor, match=pattern, count=100)

            for key in keys:
                if max_documents and doc_count >= max_documents:
                    return

                # Get document data
                data = self.redis.redis.hgetall(key)
                if not data:
                    continue

                doc_id = key.decode().split(":", 1)[1]
                content = data.get(b"content", b"").decode()
                metadata = json.loads(data.get(b"metadata", b"{}").decode())

                # Create document entity
                entity = Entity(
                    id=f"doc_{doc_id}",
                    name=metadata.get("relative_path", doc_id),
                    type=EntityType.DOCUMENT,
                    content=content[:500],  # Store excerpt
                    properties=metadata,
                    source_ids={metadata.get("source_id", "")},
                )

                # Get embedding if available
                if b"embedding" in data:
                    entity.embedding = np.frombuffer(data[b"embedding"], dtype=np.float32)

                self.entities[entity.id] = entity
                self.graph.add_node(entity.id, **asdict(entity))

                doc_count += 1

                # Extract entities from content
                await self._extract_content_entities(content, entity.id, metadata)

            if cursor == 0:
                break

    async def _extract_content_entities(
        self, content: str, doc_id: str, metadata: Dict[str, Any]
    ) -> None:
        """Extract entities from document content."""
        # Extract based on document type
        doc_type = metadata.get("file_type", "text")

        if doc_type == "code":
            await self._extract_code_entities_from_content(content, doc_id, metadata)
        elif doc_type == "markdown":
            await self._extract_markdown_entities(content, doc_id, metadata)
        else:
            await self._extract_text_entities(content, doc_id, metadata)

    async def _extract_code_entities_from_content(
        self, content: str, doc_id: str, metadata: Dict[str, Any]
    ) -> None:
        """Extract code-specific entities using pattern matching.

        Analyzes source code content to identify functions, classes, and
        import statements. Uses language-specific regular expressions to
        parse code structure and create appropriate entities.

        Supported Languages:
            - Python: functions, classes, imports (from/import)
            - JavaScript/TypeScript: functions, classes, ES6 imports
            - Future: Can be extended for other languages

        Args:
            content: Source code content to analyze.
            doc_id: Parent document ID for relationship creation.
            metadata: Document metadata including language information.

        Note:
            Uses regex patterns for simplicity. Could be enhanced with
            AST parsing for more accurate code structure analysis.
        """
        language = metadata.get("language", "unknown")

        # Simple pattern-based extraction (can be enhanced with AST)
        import re

        # Extract function definitions
        if language == "python":
            func_pattern = r"def\s+(\w+)\s*\("
            class_pattern = r"class\s+(\w+)\s*[\(:]"
            import_pattern = r"(?:from\s+[\w.]+\s+)?import\s+([\w,\s]+)"
        elif language in ["javascript", "typescript"]:
            func_pattern = r"(?:function|const|let|var)\s+(\w+)\s*(?:=\s*(?:async\s+)?\(|\()"
            class_pattern = r"class\s+(\w+)"
            import_pattern = r'import\s+.*from\s+[\'"](.+)[\'"]'
        else:
            return

        # Extract functions
        for match in re.finditer(func_pattern, content):
            func_name = match.group(1)
            entity = Entity(
                id=f"func_{doc_id}_{func_name}",
                name=func_name,
                type=EntityType.FUNCTION,
                content=f"Function {func_name} in {metadata.get('relative_path', '')}",
                properties={"language": language, "doc_id": doc_id},
                source_ids={metadata.get("source_id", "")},
            )
            self.entities[entity.id] = entity
            self.graph.add_node(entity.id, **asdict(entity))

            # Add relationship
            rel = Relationship(source_id=doc_id, target_id=entity.id, type=RelationType.CONTAINS)
            self.relationships.append(rel)
            self.graph.add_edge(doc_id, entity.id, type=rel.type.value, weight=rel.weight)

        # Extract classes
        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            entity = Entity(
                id=f"class_{doc_id}_{class_name}",
                name=class_name,
                type=EntityType.CLASS,
                content=f"Class {class_name} in {metadata.get('relative_path', '')}",
                properties={"language": language, "doc_id": doc_id},
                source_ids={metadata.get("source_id", "")},
            )
            self.entities[entity.id] = entity
            self.graph.add_node(entity.id, **asdict(entity))

            # Add relationship
            rel = Relationship(source_id=doc_id, target_id=entity.id, type=RelationType.CONTAINS)
            self.relationships.append(rel)
            self.graph.add_edge(doc_id, entity.id, type=rel.type.value, weight=rel.weight)

    async def _extract_markdown_entities(
        self, content: str, doc_id: str, metadata: Dict[str, Any]
    ) -> None:
        """Extract entities from markdown content."""
        import re

        # Extract headers as topics
        header_pattern = r"^#{1,6}\s+(.+)$"
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            header_text = match.group(1)
            entity = Entity(
                id=f"topic_{hashlib.md5(header_text.encode()).hexdigest()[:8]}",
                name=header_text,
                type=EntityType.TOPIC,
                content=header_text,
                properties={"doc_id": doc_id},
                source_ids={metadata.get("source_id", "")},
            )

            if entity.id not in self.entities:
                self.entities[entity.id] = entity
                self.graph.add_node(entity.id, **asdict(entity))

            # Add relationship
            rel = Relationship(source_id=doc_id, target_id=entity.id, type=RelationType.DESCRIBES)
            self.relationships.append(rel)
            self.graph.add_edge(doc_id, entity.id, type=rel.type.value, weight=rel.weight)

        # Extract code blocks as API references
        code_block_pattern = r"```(\w+)?\n(.*?)```"
        for match in re.finditer(code_block_pattern, content, re.DOTALL):
            language = match.group(1) or "unknown"
            code_content = match.group(2)

            if len(code_content) > 50:  # Skip very short code blocks
                entity = Entity(
                    id=f"code_{hashlib.md5(code_content.encode()).hexdigest()[:8]}",
                    name=f"Code example ({language})",
                    type=EntityType.API,
                    content=code_content[:200],
                    properties={"language": language, "doc_id": doc_id},
                    source_ids={metadata.get("source_id", "")},
                )

                if entity.id not in self.entities:
                    self.entities[entity.id] = entity
                    self.graph.add_node(entity.id, **asdict(entity))

                # Add relationship
                rel = Relationship(
                    source_id=doc_id, target_id=entity.id, type=RelationType.CONTAINS
                )
                self.relationships.append(rel)
                self.graph.add_edge(doc_id, entity.id, type=rel.type.value, weight=rel.weight)

    async def _extract_text_entities(
        self, content: str, doc_id: str, metadata: Dict[str, Any]
    ) -> None:
        """Extract entities from plain text using NLP techniques."""
        # Extract key terms (simple approach - can be enhanced with NLP)
        import re

        # Extract capitalized terms (potential entities)
        capitalized_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
        seen_terms = set()

        for match in re.finditer(capitalized_pattern, content):
            term = match.group(0)
            if len(term) > 3 and term not in seen_terms:
                seen_terms.add(term)

                # Determine entity type based on context
                entity_type = EntityType.TERM
                if any(tech in term.lower() for tech in ["api", "sdk", "framework", "library"]):
                    entity_type = EntityType.TECHNOLOGY
                elif any(org in term.lower() for org in ["inc", "corp", "company", "foundation"]):
                    entity_type = EntityType.ORGANIZATION

                entity = Entity(
                    id=f"term_{hashlib.md5(term.encode()).hexdigest()[:8]}",
                    name=term,
                    type=entity_type,
                    content=term,
                    properties={"doc_id": doc_id},
                    source_ids={metadata.get("source_id", "")},
                )

                if entity.id not in self.entities:
                    self.entities[entity.id] = entity
                    self.graph.add_node(entity.id, **asdict(entity))

                # Add relationship
                rel = Relationship(
                    source_id=doc_id, target_id=entity.id, type=RelationType.REFERENCES
                )
                self.relationships.append(rel)
                self.graph.add_edge(doc_id, entity.id, type=rel.type.value, weight=rel.weight)

    async def _extract_code_entities(self, source_id: Optional[str] = None) -> None:
        """Extract code-specific entities from indexed code files."""
        # This is handled in _extract_code_entities_from_content
        pass

    async def _extract_conceptual_entities(self, source_id: Optional[str] = None) -> None:
        """Extract high-level conceptual entities."""
        # Scan for concept-level documents
        pattern = f"concept:{source_id}*" if source_id else "concept:*"
        cursor = 0

        while True:
            cursor, keys = self.redis.redis.scan(cursor, match=pattern, count=100)

            for key in keys:
                data = self.redis.redis.hgetall(key)
                if not data:
                    continue

                concept_id = key.decode().split(":", 1)[1]
                content = data.get(b"content", b"").decode()
                metadata = json.loads(data.get(b"metadata", b"{}").decode())

                # Create concept entity
                entity = Entity(
                    id=f"concept_{concept_id}",
                    name=metadata.get("section_title", concept_id),
                    type=EntityType.CONCEPT,
                    content=content,
                    properties=metadata,
                    source_ids={metadata.get("source_id", "")},
                )

                # Get embedding if available
                if b"embedding" in data:
                    entity.embedding = np.frombuffer(data[b"embedding"], dtype=np.float32)

                self.entities[entity.id] = entity
                self.graph.add_node(entity.id, **asdict(entity))

            if cursor == 0:
                break

    async def _build_structural_relationships(self) -> None:
        """Build structural relationships (parent-child, contains, etc.)."""
        # Already built during entity extraction
        pass

    async def _build_semantic_relationships(self) -> None:
        """Build semantic relationships using vector embedding similarity.

        Calculates cosine similarity between entity embeddings and creates
        SIMILAR_TO relationships for highly similar entities. This captures
        semantic relationships that aren't apparent from structural analysis.

        Process:
        1. Filter entities that have embeddings
        2. Calculate pairwise cosine similarities
        3. Create relationships for similarities above 0.8 threshold
        4. Store similarity scores as relationship weights

        Note:
            Computationally intensive for large entity sets (O(n²)).
            Consider sampling or clustering for very large graphs.
        """
        # Get entities with embeddings
        entities_with_embeddings = [e for e in self.entities.values() if e.embedding is not None]

        if len(entities_with_embeddings) < 2:
            return

        # Calculate similarity between entities
        for i, entity1 in enumerate(entities_with_embeddings):
            for entity2 in entities_with_embeddings[i + 1 :]:
                if entity1.embedding is not None and entity2.embedding is not None:
                    # Calculate cosine similarity
                    similarity = np.dot(entity1.embedding, entity2.embedding) / (
                        np.linalg.norm(entity1.embedding) * np.linalg.norm(entity2.embedding)
                    )

                    # Add relationship if similarity is high
                    if similarity > 0.8:
                        rel = Relationship(
                            source_id=entity1.id,
                            target_id=entity2.id,
                            type=RelationType.SIMILAR_TO,
                            weight=float(similarity),
                            properties={"similarity": float(similarity)},
                        )
                        self.relationships.append(rel)
                        self.graph.add_edge(
                            entity1.id, entity2.id, type=rel.type.value, weight=rel.weight
                        )

    async def _build_code_relationships(self) -> None:
        """Build code-specific relationships (calls, imports, etc.)."""
        # Analyze function and class relationships
        functions = [e for e in self.entities.values() if e.type == EntityType.FUNCTION]
        classes = [e for e in self.entities.values() if e.type == EntityType.CLASS]

        # Build inheritance relationships (simplified)
        for cls in classes:
            # Check if class name suggests inheritance
            if "Base" in cls.name or "Abstract" in cls.name:
                for other_cls in classes:
                    if other_cls.id != cls.id and cls.name in other_cls.content:
                        rel = Relationship(
                            source_id=other_cls.id, target_id=cls.id, type=RelationType.EXTENDS
                        )
                        self.relationships.append(rel)
                        self.graph.add_edge(
                            other_cls.id, cls.id, type=rel.type.value, weight=rel.weight
                        )

    async def _store_graph(self) -> None:
        """Store knowledge graph in Redis."""
        # Store entities
        for entity in self.entities.values():
            entity_key = f"kg_entity:{entity.id}"
            entity_data = {
                "name": entity.name,
                "type": entity.type.value,
                "content": entity.content,
                "properties": json.dumps(entity.properties),
                "source_ids": json.dumps(list(entity.source_ids)),
            }

            if entity.embedding is not None:
                entity_data["embedding"] = entity.embedding.tobytes()

            self.redis.redis.hset(entity_key, mapping=entity_data)

        # Store relationships
        for rel in self.relationships:
            rel_key = f"kg_rel:{rel.source_id}:{rel.target_id}:{rel.type.value}"
            rel_data = {
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "type": rel.type.value,
                "weight": rel.weight,
                "properties": json.dumps(rel.properties),
            }
            self.redis.redis.hset(rel_key, mapping=rel_data)

        # Store graph metadata
        graph_meta_key = "kg_metadata"
        self.redis.redis.hset(
            graph_meta_key,
            mapping={
                "entity_count": len(self.entities),
                "relationship_count": len(self.relationships),
                "last_updated": json.dumps({"timestamp": time.time()}),
            },
        )

    async def query_subgraph(
        self, query: str, max_depth: int = 2, max_entities: int = 20
    ) -> KnowledgeSubgraph:
        """Query knowledge graph to retrieve relevant subgraph for a specific query.

        Performs semantic search to find the most relevant entities for the query,
        then traverses the graph to build a focused subgraph containing related
        entities and their relationships. Uses breadth-first search with depth
        limiting to control subgraph size.

        Query Process:
        1. Generate embedding for input query
        2. Find most relevant entities using vector similarity
        3. Perform BFS traversal from central entities
        4. Collect connected entities within max_depth hops
        5. Extract relationships between included entities
        6. Return structured subgraph with metadata

        Args:
            query: Natural language query for finding relevant knowledge.
            max_depth: Maximum graph traversal depth from central entities.
                Higher values include more distant relationships.
            max_entities: Maximum number of entities to include in subgraph.
                Controls computational complexity and context size.

        Returns:
            KnowledgeSubgraph containing:
            - entities: List of relevant Entity objects
            - relationships: List of relationships between entities
            - central_entities: IDs of most relevant entities to query
            - metadata: Query context and subgraph statistics

        Example:
            Query for authentication-related information:

            >>> subgraph = await kg_builder.query_subgraph(
            ...     query="user authentication and security",
            ...     max_depth=3,
            ...     max_entities=15
            ... )
            >>>
            >>> # Analyze central entities
            >>> central = [e for e in subgraph.entities if e.id in subgraph.central_entities]
            >>> for entity in central:
            ...     print(f"Central: {entity.name} ({entity.type.value})")
            >>>
            >>> # Check relationship patterns
            >>> rel_counts = {}
            >>> for rel in subgraph.relationships:
            ...     rel_counts[rel.type.value] = rel_counts.get(rel.type.value, 0) + 1
            >>> print(f"Relationship distribution: {rel_counts}")
        """
        # Get query embedding
        query_embedding = await self.embeddings.get_embedding(query)

        # Find most relevant entities using vector search
        central_entities = await self._find_relevant_entities(query_embedding, k=5)

        # Traverse graph to get connected entities
        subgraph_entities = set(central_entities)
        subgraph_relationships = []

        # BFS traversal
        queue = [(entity_id, 0) for entity_id in central_entities]
        visited = set()

        while queue and len(subgraph_entities) < max_entities:
            entity_id, depth = queue.pop(0)

            if entity_id in visited or depth >= max_depth:
                continue

            visited.add(entity_id)

            # Get connected entities
            if entity_id in self.graph:
                # Outgoing edges
                for neighbor in self.graph.neighbors(entity_id):
                    if neighbor not in subgraph_entities:
                        subgraph_entities.add(neighbor)
                        queue.append((neighbor, depth + 1))

                    # Add relationship
                    edge_data = self.graph.get_edge_data(entity_id, neighbor)
                    if edge_data:
                        for key, data in edge_data.items():
                            rel = Relationship(
                                source_id=entity_id,
                                target_id=neighbor,
                                type=RelationType(data.get("type", "relates_to")),
                                weight=data.get("weight", 1.0),
                            )
                            subgraph_relationships.append(rel)

                # Incoming edges
                for predecessor in self.graph.predecessors(entity_id):
                    if predecessor not in subgraph_entities:
                        subgraph_entities.add(predecessor)
                        queue.append((predecessor, depth + 1))

                    # Add relationship
                    edge_data = self.graph.get_edge_data(predecessor, entity_id)
                    if edge_data:
                        for key, data in edge_data.items():
                            rel = Relationship(
                                source_id=predecessor,
                                target_id=entity_id,
                                type=RelationType(data.get("type", "relates_to")),
                                weight=data.get("weight", 1.0),
                            )
                            subgraph_relationships.append(rel)

        # Get entity objects
        result_entities = [self.entities[eid] for eid in subgraph_entities if eid in self.entities]

        return KnowledgeSubgraph(
            entities=result_entities[:max_entities],
            relationships=subgraph_relationships,
            central_entities=central_entities,
            metadata={
                "query": query,
                "depth": max_depth,
                "total_entities": len(result_entities),
                "total_relationships": len(subgraph_relationships),
            },
        )

    async def _find_relevant_entities(self, query_embedding: np.ndarray, k: int = 5) -> List[str]:
        """Find k most relevant entities using vector similarity search.

        Searches through all stored entity embeddings to find those most
        similar to the query embedding. Used as starting points for
        subgraph traversal in query processing.

        Args:
            query_embedding: Query vector for similarity comparison.
            k: Number of top similar entities to return.

        Returns:
            List of entity IDs ranked by similarity to query.

        Note:
            Scans all entities in Redis, which may be slow for large graphs.
            Could be optimized with dedicated vector search index.
        """
        relevant = []

        # Search in stored entities
        pattern = "kg_entity:*"
        cursor = 0
        similarities = []

        while True:
            cursor, keys = self.redis.redis.scan(cursor, match=pattern, count=100)

            for key in keys:
                data = self.redis.redis.hgetall(key)
                if b"embedding" in data:
                    entity_id = key.decode().split(":", 1)[1]
                    embedding = np.frombuffer(data[b"embedding"], dtype=np.float32)

                    # Calculate similarity
                    similarity = np.dot(query_embedding, embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                    )

                    similarities.append((entity_id, similarity))

            if cursor == 0:
                break

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [entity_id for entity_id, _ in similarities[:k]]

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph structure and content.

        Provides detailed metrics about graph composition, connectivity, and
        distribution of entity and relationship types. Useful for monitoring
        graph growth, analyzing knowledge coverage, and optimizing performance.

        Returns:
            Dictionary containing:
            - entity_count: Total number of entities in the graph
            - relationship_count: Total number of relationships
            - entity_types: Distribution of entity types with counts
            - relationship_types: Distribution of relationship types with counts
            - connected_components: Number of weakly connected components
            - density: Graph density (0-1, higher = more connected)

        Example:
            >>> stats = kg_builder.get_graph_stats()
            >>> print(f"Graph Overview:")
            >>> print(f"  Entities: {stats['entity_count']}")
            >>> print(f"  Relationships: {stats['relationship_count']}")
            >>> print(f"  Density: {stats['density']:.3f}")
            >>> print(f"  Components: {stats['connected_components']}")
            >>>
            >>> # Analyze entity distribution
            >>> print("\nEntity Types:")
            >>> for entity_type, count in stats['entity_types'].items():
            ...     print(f"  {entity_type}: {count}")
            >>>
            >>> # Analyze relationship patterns
            >>> print("\nTop Relationship Types:")
            >>> sorted_rels = sorted(stats['relationship_types'].items(),
            ...                      key=lambda x: x[1], reverse=True)
            >>> for rel_type, count in sorted_rels[:5]:
            ...     print(f"  {rel_type}: {count}")
        """
        return {
            "entity_count": len(self.entities),
            "relationship_count": len(self.relationships),
            "entity_types": dict(
                defaultdict(
                    int,
                    {
                        entity.type.value: sum(
                            1 for e in self.entities.values() if e.type == entity.type
                        )
                        for entity in self.entities.values()
                    },
                )
            ),
            "relationship_types": dict(
                defaultdict(
                    int,
                    {
                        rel.type.value: sum(1 for r in self.relationships if r.type == rel.type)
                        for rel in self.relationships
                    },
                )
            ),
            "connected_components": nx.number_weakly_connected_components(self.graph),
            "density": nx.density(self.graph),
        }

    async def discover_patterns(self, min_support: float = 0.1) -> List[Dict[str, Any]]:
        """Discover common structural and semantic patterns in the knowledge graph.

        Analyzes the knowledge graph to identify recurring patterns, hub entities,
        and community structures. This provides insights into knowledge organization,
        identifies important entities, and reveals structural characteristics.

        Pattern Types Discovered:
        - Entity-Relationship Patterns: Common triplet patterns (A -rel-> B)
        - Hub Entities: Highly connected nodes (top 10% by degree)
        - Communities: Densely connected subgroups of entities
        - Support Analysis: Statistical significance of patterns

        Args:
            min_support: Minimum support threshold (0.0-1.0) for including patterns.
                Support = pattern_count / total_relationships. Higher values
                return only most common patterns.

        Returns:
            List of pattern dictionaries containing:
            - pattern: Pattern description or type identifier
            - count: Absolute occurrence count
            - support: Relative frequency (0.0-1.0)
            - Additional pattern-specific fields (entities, communities, etc.)

        Example:
            Discover and analyze graph patterns:

            >>> patterns = await kg_builder.discover_patterns(min_support=0.05)
            >>>
            >>> # Analyze relationship patterns
            >>> rel_patterns = [p for p in patterns if '->' in p.get('pattern', '')]
            >>> for pattern in rel_patterns[:3]:
            ...     print(f"Pattern: {pattern['pattern']}")
            ...     print(f"  Count: {pattern['count']} (support: {pattern['support']:.1%})")
            >>>
            >>> # Find hub entities
            >>> hub_pattern = next((p for p in patterns if p['pattern'] == 'hub_entities'), None)
            >>> if hub_pattern:
            ...     print(f"\nHub Entities (threshold: {hub_pattern['threshold']:.1f}):")
            ...     for hub in hub_pattern['entities'][:5]:
            ...         print(f"  {hub['entity_name']} ({hub['type']}) - degree: {hub['degree']}")
            >>>
            >>> # Analyze communities
            >>> comm_pattern = next((p for p in patterns if p['pattern'] == 'communities'), None)
            >>> if comm_pattern:
            ...     print(f"\nFound {comm_pattern['count']} communities")
            ...     for comm_id, members in list(comm_pattern['communities'].items())[:3]:
            ...         print(f"  Community {comm_id}: {len(members)} members")

        Note:
            Community detection requires the 'python-louvain' package.
            If not available, community patterns are skipped silently.
        """
        patterns = []

        # Find common subgraphs
        entity_type_pairs = defaultdict(int)
        for rel in self.relationships:
            if rel.source_id in self.entities and rel.target_id in self.entities:
                source_type = self.entities[rel.source_id].type.value
                target_type = self.entities[rel.target_id].type.value
                rel_type = rel.type.value

                pattern = f"{source_type} -{rel_type}-> {target_type}"
                entity_type_pairs[pattern] += 1

        # Filter by minimum support
        total_relationships = len(self.relationships)
        for pattern, count in entity_type_pairs.items():
            support = count / total_relationships if total_relationships > 0 else 0
            if support >= min_support:
                patterns.append(
                    {
                        "pattern": pattern,
                        "count": count,
                        "support": support,
                    }
                )

        # Find hub entities (highly connected)
        node_degrees = dict(self.graph.degree())
        hub_threshold = np.percentile(list(node_degrees.values()), 90) if node_degrees else 0

        hubs = [
            {
                "entity_id": node,
                "entity_name": self.entities[node].name if node in self.entities else node,
                "degree": degree,
                "type": self.entities[node].type.value if node in self.entities else "unknown",
            }
            for node, degree in node_degrees.items()
            if degree >= hub_threshold
        ]

        if hubs:
            patterns.append(
                {
                    "pattern": "hub_entities",
                    "entities": hubs,
                    "threshold": hub_threshold,
                }
            )

        # Find clusters (communities)
        if len(self.graph) > 10:
            try:
                import community

                partition = community.best_partition(self.graph.to_undirected())

                communities = defaultdict(list)
                for node, comm_id in partition.items():
                    if node in self.entities:
                        communities[comm_id].append(
                            {
                                "id": node,
                                "name": self.entities[node].name,
                                "type": self.entities[node].type.value,
                            }
                        )

                patterns.append(
                    {
                        "pattern": "communities",
                        "count": len(communities),
                        "communities": dict(communities),
                    }
                )
            except:
                pass  # Community detection not available

        return patterns


import time  # Add this import at the top of the file
