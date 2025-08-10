"""
Knowledge graph builder for knowledge discovery.
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
    """Types of entities in the knowledge graph."""
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
    """Types of relationships in the knowledge graph."""
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
    """Entity in the knowledge graph."""
    id: str
    name: str
    type: EntityType
    content: str = ""
    embedding: Optional[np.ndarray] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    source_ids: Set[str] = field(default_factory=set)


@dataclass
class Relationship:
    """Relationship between entities."""
    source_id: str
    target_id: str
    type: RelationType
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeSubgraph:
    """Subgraph for a specific query or context."""
    entities: List[Entity]
    relationships: List[Relationship]
    central_entities: List[str]  # Most relevant entity IDs
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraphBuilder:
    """Builds and manages knowledge graph from indexed documents."""
    
    def __init__(
        self,
        redis_store: RedisVectorStore,
        embedding_manager: EmbeddingManager
    ):
        self.redis = redis_store
        self.embeddings = embedding_manager
        self.graph = nx.MultiDiGraph()  # Multi-directed graph for multiple relationship types
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        
    async def build_from_documents(
        self,
        source_id: Optional[str] = None,
        max_documents: Optional[int] = None
    ) -> None:
        """Build knowledge graph from indexed documents."""
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
        
        logger.info(f"Built knowledge graph with {len(self.entities)} entities and {len(self.relationships)} relationships")
    
    async def _extract_document_entities(
        self,
        source_id: Optional[str] = None,
        max_documents: Optional[int] = None
    ) -> None:
        """Extract document-level entities."""
        # Scan for documents
        pattern = f"chunk:{source_id}*" if source_id else "chunk:*"
        cursor = 0
        doc_count = 0
        
        while True:
            cursor, keys = await self.redis.redis.scan(
                cursor,
                match=pattern,
                count=100
            )
            
            for key in keys:
                if max_documents and doc_count >= max_documents:
                    return
                
                # Get document data
                data = await self.redis.redis.hgetall(key)
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
                    source_ids={metadata.get("source_id", "")}
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
        self,
        content: str,
        doc_id: str,
        metadata: Dict[str, Any]
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
        self,
        content: str,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Extract code entities (functions, classes, imports)."""
        language = metadata.get("language", "unknown")
        
        # Simple pattern-based extraction (can be enhanced with AST)
        import re
        
        # Extract function definitions
        if language == "python":
            func_pattern = r'def\s+(\w+)\s*\('
            class_pattern = r'class\s+(\w+)\s*[\(:]'
            import_pattern = r'(?:from\s+[\w.]+\s+)?import\s+([\w,\s]+)'
        elif language in ["javascript", "typescript"]:
            func_pattern = r'(?:function|const|let|var)\s+(\w+)\s*(?:=\s*(?:async\s+)?\(|\()'
            class_pattern = r'class\s+(\w+)'
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
                source_ids={metadata.get("source_id", "")}
            )
            self.entities[entity.id] = entity
            self.graph.add_node(entity.id, **asdict(entity))
            
            # Add relationship
            rel = Relationship(
                source_id=doc_id,
                target_id=entity.id,
                type=RelationType.CONTAINS
            )
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
                source_ids={metadata.get("source_id", "")}
            )
            self.entities[entity.id] = entity
            self.graph.add_node(entity.id, **asdict(entity))
            
            # Add relationship
            rel = Relationship(
                source_id=doc_id,
                target_id=entity.id,
                type=RelationType.CONTAINS
            )
            self.relationships.append(rel)
            self.graph.add_edge(doc_id, entity.id, type=rel.type.value, weight=rel.weight)
    
    async def _extract_markdown_entities(
        self,
        content: str,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Extract entities from markdown content."""
        import re
        
        # Extract headers as topics
        header_pattern = r'^#{1,6}\s+(.+)$'
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            header_text = match.group(1)
            entity = Entity(
                id=f"topic_{hashlib.md5(header_text.encode()).hexdigest()[:8]}",
                name=header_text,
                type=EntityType.TOPIC,
                content=header_text,
                properties={"doc_id": doc_id},
                source_ids={metadata.get("source_id", "")}
            )
            
            if entity.id not in self.entities:
                self.entities[entity.id] = entity
                self.graph.add_node(entity.id, **asdict(entity))
            
            # Add relationship
            rel = Relationship(
                source_id=doc_id,
                target_id=entity.id,
                type=RelationType.DESCRIBES
            )
            self.relationships.append(rel)
            self.graph.add_edge(doc_id, entity.id, type=rel.type.value, weight=rel.weight)
        
        # Extract code blocks as API references
        code_block_pattern = r'```(\w+)?\n(.*?)```'
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
                    source_ids={metadata.get("source_id", "")}
                )
                
                if entity.id not in self.entities:
                    self.entities[entity.id] = entity
                    self.graph.add_node(entity.id, **asdict(entity))
                
                # Add relationship
                rel = Relationship(
                    source_id=doc_id,
                    target_id=entity.id,
                    type=RelationType.CONTAINS
                )
                self.relationships.append(rel)
                self.graph.add_edge(doc_id, entity.id, type=rel.type.value, weight=rel.weight)
    
    async def _extract_text_entities(
        self,
        content: str,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Extract entities from plain text using NLP techniques."""
        # Extract key terms (simple approach - can be enhanced with NLP)
        import re
        
        # Extract capitalized terms (potential entities)
        capitalized_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
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
                    source_ids={metadata.get("source_id", "")}
                )
                
                if entity.id not in self.entities:
                    self.entities[entity.id] = entity
                    self.graph.add_node(entity.id, **asdict(entity))
                
                # Add relationship
                rel = Relationship(
                    source_id=doc_id,
                    target_id=entity.id,
                    type=RelationType.REFERENCES
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
            cursor, keys = await self.redis.redis.scan(
                cursor,
                match=pattern,
                count=100
            )
            
            for key in keys:
                data = await self.redis.redis.hgetall(key)
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
                    source_ids={metadata.get("source_id", "")}
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
        """Build semantic relationships based on similarity."""
        # Get entities with embeddings
        entities_with_embeddings = [
            e for e in self.entities.values() 
            if e.embedding is not None
        ]
        
        if len(entities_with_embeddings) < 2:
            return
        
        # Calculate similarity between entities
        for i, entity1 in enumerate(entities_with_embeddings):
            for entity2 in entities_with_embeddings[i+1:]:
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
                            properties={"similarity": float(similarity)}
                        )
                        self.relationships.append(rel)
                        self.graph.add_edge(
                            entity1.id,
                            entity2.id,
                            type=rel.type.value,
                            weight=rel.weight
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
                            source_id=other_cls.id,
                            target_id=cls.id,
                            type=RelationType.EXTENDS
                        )
                        self.relationships.append(rel)
                        self.graph.add_edge(
                            other_cls.id,
                            cls.id,
                            type=rel.type.value,
                            weight=rel.weight
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
            
            await self.redis.redis.hset(entity_key, mapping=entity_data)
        
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
            await self.redis.redis.hset(rel_key, mapping=rel_data)
        
        # Store graph metadata
        graph_meta_key = "kg_metadata"
        await self.redis.redis.hset(graph_meta_key, mapping={
            "entity_count": len(self.entities),
            "relationship_count": len(self.relationships),
            "last_updated": json.dumps({"timestamp": time.time()}),
        })
    
    async def query_subgraph(
        self,
        query: str,
        max_depth: int = 2,
        max_entities: int = 20
    ) -> KnowledgeSubgraph:
        """
        Query knowledge graph to get relevant subgraph for a query.
        
        Args:
            query: User query
            max_depth: Maximum traversal depth from central entities
            max_entities: Maximum number of entities to return
            
        Returns:
            KnowledgeSubgraph with relevant entities and relationships
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
                                weight=data.get("weight", 1.0)
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
                                weight=data.get("weight", 1.0)
                            )
                            subgraph_relationships.append(rel)
        
        # Get entity objects
        result_entities = [
            self.entities[eid] for eid in subgraph_entities 
            if eid in self.entities
        ]
        
        return KnowledgeSubgraph(
            entities=result_entities[:max_entities],
            relationships=subgraph_relationships,
            central_entities=central_entities,
            metadata={
                "query": query,
                "depth": max_depth,
                "total_entities": len(result_entities),
                "total_relationships": len(subgraph_relationships),
            }
        )
    
    async def _find_relevant_entities(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[str]:
        """Find most relevant entities using vector similarity."""
        relevant = []
        
        # Search in stored entities
        pattern = "kg_entity:*"
        cursor = 0
        similarities = []
        
        while True:
            cursor, keys = await self.redis.redis.scan(
                cursor,
                match=pattern,
                count=100
            )
            
            for key in keys:
                data = await self.redis.redis.hgetall(key)
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
        """Get knowledge graph statistics."""
        return {
            "entity_count": len(self.entities),
            "relationship_count": len(self.relationships),
            "entity_types": dict(defaultdict(int, {
                entity.type.value: sum(1 for e in self.entities.values() if e.type == entity.type)
                for entity in self.entities.values()
            })),
            "relationship_types": dict(defaultdict(int, {
                rel.type.value: sum(1 for r in self.relationships if r.type == rel.type)
                for rel in self.relationships
            })),
            "connected_components": nx.number_weakly_connected_components(self.graph),
            "density": nx.density(self.graph),
        }
    
    async def discover_patterns(self, min_support: float = 0.1) -> List[Dict[str, Any]]:
        """
        Discover common patterns in the knowledge graph.
        
        Args:
            min_support: Minimum support threshold for patterns
            
        Returns:
            List of discovered patterns
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
                patterns.append({
                    "pattern": pattern,
                    "count": count,
                    "support": support,
                })
        
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
            patterns.append({
                "pattern": "hub_entities",
                "entities": hubs,
                "threshold": hub_threshold,
            })
        
        # Find clusters (communities)
        if len(self.graph) > 10:
            try:
                import community
                partition = community.best_partition(self.graph.to_undirected())
                
                communities = defaultdict(list)
                for node, comm_id in partition.items():
                    if node in self.entities:
                        communities[comm_id].append({
                            "id": node,
                            "name": self.entities[node].name,
                            "type": self.entities[node].type.value,
                        })
                
                patterns.append({
                    "pattern": "communities",
                    "count": len(communities),
                    "communities": dict(communities),
                })
            except:
                pass  # Community detection not available
        
        return patterns


import time  # Add this import at the top of the file