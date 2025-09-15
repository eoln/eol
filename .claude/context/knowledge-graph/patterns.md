# Knowledge Graph Implementation Patterns

## Discovered Patterns

### Entity Extraction Patterns

#### Code Entity Extraction

```python
# Pattern: AST-based entity extraction for code
class EntityExtractor:
    def extract_from_code(self, content: str, language: str) -> List[Entity]:
        """Extract entities using AST parsing"""
        if language == "python":
            return self._extract_python_entities(content)
        elif language in ["javascript", "typescript"]:
            return self._extract_js_entities(content)
```

**Entity Types to Extract:**

- **Classes**: Name, methods, inheritance, docstring
- **Functions**: Name, parameters, return types, docstring
- **Variables**: Global constants, configuration values
- **Modules**: Import relationships, exports
- **Decorators/Annotations**: Special markers and metadata

#### Document Entity Extraction

```python
def extract_from_markdown(self, content: str) -> List[Entity]:
    """Extract entities from markdown structure"""
    # Parse headers, links, code blocks
```

**Document Entities:**

- Headers: Section titles and hierarchy
- Concepts: Key terms and definitions
- References: Links, citations, cross-references
- Code Blocks: Embedded examples
- Lists: Enumerated concepts and steps

#### Data Entity Extraction

```python
def extract_from_data(self, content: dict, format: str) -> List[Entity]:
    """Extract entities from structured data"""
    # Parse JSON/YAML schemas
```

**Data Entities:**

- Schema Definitions: Structure and types
- Configuration Keys: Settings and parameters
- API Endpoints: Routes and methods
- Data Models: Object structures

### Integration Patterns

#### Document Processing Integration

```python
# Pattern: Augment existing document processor
async def process_document(self, file_path: Path) -> ProcessedDocument:
    # Existing processing...

    # NEW: Extract entities
    entities = await self.entity_extractor.extract(
        content=content,
        file_type=file_type,
        metadata=metadata
    )

    # NEW: Add entities to document
    document.entities = entities

    return document
```

#### Indexer Integration

```python
# Pattern: Hook into indexing pipeline
async def index_file(self, file_path: Path, source_id: str) -> IndexResult:
    # Process document
    document = await self.processor.process_document(file_path)

    # Store chunks (existing)
    await self._store_chunks(document.chunks)

    # NEW: Build knowledge graph
    await self._update_knowledge_graph(document, source_id)

async def _update_knowledge_graph(self, document: ProcessedDocument, source_id: str):
    """Update knowledge graph with document entities"""
    # Add entities
    for entity in document.entities:
        await self.kg_builder.add_entity(
            entity_id=f"{source_id}_{entity.id}",
            entity_type=entity.type,
            properties={
                "name": entity.name,
                "file_path": str(document.file_path),
                "source_id": source_id,
                **entity.metadata
            }
        )

    # Discover relationships
    await self._discover_relationships(document.entities, source_id)
```

### Relationship Discovery Patterns

#### Structural Relationship Discovery

```python
class RelationshipDiscovery:
    async def discover_code_relationships(self, entities: List[Entity]) -> List[Relationship]:
        """Discover relationships in code"""
        relationships = []

        # Import relationships
        for entity in entities:
            if entity.type == "import":
                relationships.append(Relationship(
                    source=entity.file,
                    target=entity.imported_module,
                    type="IMPORTS"
                ))

        # Inheritance relationships
        for entity in entities:
            if entity.type == "class" and entity.base_classes:
                for base in entity.base_classes:
                    relationships.append(Relationship(
                        source=entity.id,
                        target=base,
                        type="INHERITS"
                    ))

        return relationships
```

**Relationship Types:**

1. **Structural**: CONTAINS, IMPORTS, INHERITS, IMPLEMENTS, REFERENCES
2. **Semantic**: SIMILAR_TO, RELATED_TO, DEPENDS_ON, USES
3. **Hierarchical**: PARENT_OF, SECTION_OF, PART_OF

#### Semantic Relationship Discovery

```python
async def discover_semantic_relationships(self, entities: List[Entity]) -> List[Relationship]:
    """Discover relationships using embeddings"""
    # Generate embeddings for entities
    embeddings = await self.embedding_manager.get_embeddings(
        [e.description for e in entities]
    )

    # Find similar entities
    similarity_threshold = 0.8
    for i, entity1 in enumerate(entities):
        for j, entity2 in enumerate(entities[i+1:], i+1):
            similarity = cosine_similarity(embeddings[i], embeddings[j])
            if similarity > similarity_threshold:
                relationships.append(Relationship(
                    source=entity1.id,
                    target=entity2.id,
                    type="SIMILAR_TO",
                    weight=similarity
                ))
```

### Parallel Processing Patterns

#### Batch Entity Processing

```python
class ParallelKnowledgeGraphBuilder:
    async def process_batch(self, documents: List[ProcessedDocument]):
        """Process documents in parallel for KG building"""
        # Collect all entities
        all_entities = []
        entity_batches = []

        for doc in documents:
            entities = await self.extract_entities(doc)
            all_entities.extend(entities)
            entity_batches.append((doc.source_id, entities))

        # Batch store entities
        await self.batch_store_entities(all_entities)

        # Discover relationships in parallel
        tasks = [
            self.discover_relationships_for_batch(batch)
            for batch in entity_batches
        ]
        relationships = await asyncio.gather(*tasks)

        # Batch store relationships
        await self.batch_store_relationships(relationships)
```

### Storage Patterns

#### Redis Storage Schema

```python
# Entity storage
entity:{source_id}:{entity_id} = {
    "type": "class|function|concept",
    "name": "EntityName",
    "properties": {...},
    "embedding": [...],
    "created_at": timestamp,
    "updated_at": timestamp
}

# Relationship storage
relationship:{source_id}:{rel_id} = {
    "source": "entity_id",
    "target": "entity_id",
    "type": "IMPORTS|CONTAINS|SIMILAR_TO",
    "weight": 0.95,
    "properties": {...}
}

# Graph metadata
graph:metadata:{source_id} = {
    "entity_count": 1000,
    "relationship_count": 5000,
    "last_updated": timestamp,
    "schema_version": "1.0"
}
```

#### Incremental Update Pattern

```python
class IncrementalGraphUpdater:
    async def update_for_file(self, file_path: Path, source_id: str):
        """Update graph for a single file change"""
        # Remove old entities for this file
        old_entities = await self.get_entities_for_file(file_path)
        await self.remove_entities(old_entities)

        # Process new version
        document = await self.processor.process_document(file_path)
        new_entities = await self.extract_entities(document)

        # Add new entities
        await self.add_entities(new_entities)

        # Update relationships
        await self.update_relationships_for_entities(new_entities)
```

### Advanced Query Patterns

#### Multi-Strategy Querying

```python
async def enhanced_query(self, query: str, strategy: str = "hybrid"):
    """Enhanced querying with multiple strategies"""

    if strategy == "hybrid":
        # Combine multiple approaches
        keyword_results = await self.keyword_search(query)
        semantic_results = await self.semantic_search(query)
        graph_results = await self.graph_traversal(query)

        # Merge and rank results
        return self.merge_results(keyword_results, semantic_results, graph_results)

    elif strategy == "path_finding":
        # Find paths between concepts
        source = self.extract_source_concept(query)
        target = self.extract_target_concept(query)
        return await self.find_paths(source, target)

    elif strategy == "neighborhood":
        # Explore entity neighborhood
        entity = self.find_central_entity(query)
        return await self.get_neighborhood(entity, depth=2)
```

#### Community Detection Pattern

```python
async def detect_communities(self) -> List[Community]:
    """Detect communities in the knowledge graph"""
    # Load graph into NetworkX
    G = await self.load_graph_to_networkx()

    # Apply community detection
    communities = nx.community.louvain_communities(G)

    # Analyze communities
    return [
        Community(
            id=f"community_{i}",
            entities=list(community),
            cohesion=self.calculate_cohesion(community, G),
            description=self.generate_description(community)
        )
        for i, community in enumerate(communities)
    ]
```

### Performance Optimization Patterns

#### Caching Strategy

- Cache entity embeddings to avoid regeneration
- Cache frequently accessed subgraphs
- Use Redis TTL for automatic cache expiration

#### Batch Operations

- Process documents in batches for efficiency
- Use Redis pipelines for bulk operations
- Parallelize relationship discovery

#### Index Optimization

- Create compound indices for complex queries
- Use hierarchical indices for multi-level retrieval
- Implement incremental index updates

### Testing Patterns

```python
class TestKnowledgeGraph:
    async def test_entity_extraction(self):
        """Test entity extraction from various file types"""
        # Test Python entities
        python_entities = await extractor.extract_from_code(python_code, "python")
        assert any(e.type == "class" for e in python_entities)

        # Test Markdown entities
        md_entities = await extractor.extract_from_markdown(markdown_content)
        assert any(e.type == "header" for e in md_entities)

    async def test_relationship_discovery(self):
        """Test relationship discovery algorithms"""
        entities = [...]
        relationships = await discoverer.discover_relationships(entities)
        assert any(r.type == "IMPORTS" for r in relationships)

    async def test_incremental_updates(self):
        """Test incremental graph updates"""
        # Initial indexing
        await indexer.index_file(file_path, source_id)
        initial_count = await kg.get_entity_count()

        # Update file
        await updater.update_for_file(file_path, source_id)
        new_count = await kg.get_entity_count()
        assert new_count >= initial_count
```

## Implementation Checklist

### Phase 1: Entity Extraction (Week 1-2)

- [ ] Implement Python entity extractor using AST
- [ ] Implement JavaScript/TypeScript extractor
- [ ] Implement Markdown entity extractor
- [ ] Implement JSON/YAML entity extractor
- [ ] Unit tests for each extractor

### Phase 2: Pipeline Integration (Week 3-4)

- [ ] Modify document_processor.py to extract entities
- [ ] Update indexer.py to build knowledge graph
- [ ] Integrate with parallel_indexer.py
- [ ] Add file watcher integration
- [ ] Integration tests

### Phase 3: Relationship Discovery (Week 5-6)

- [ ] Implement structural relationship discovery
- [ ] Implement semantic relationship discovery using embeddings
- [ ] Add relationship weighting
- [ ] Create relationship persistence layer
- [ ] Performance optimization

### Phase 4: Query & Advanced Features (Week 7-8)

- [ ] Implement multi-strategy querying
- [ ] Add community detection
- [ ] Implement pattern mining
- [ ] Create visualization tools
- [ ] Documentation and examples

## Success Metrics

1. **Coverage**: >90% of code entities extracted
2. **Accuracy**: >85% relationship accuracy
3. **Performance**: <100ms query response time
4. **Scale**: Handle 100k+ entities efficiently
5. **Incremental**: <1s update time per file change

## Integration Points

### Existing Components

- `document_processor.py`: Add entity extraction
- `indexer.py`: Integrate KG building
- `parallel_indexer.py`: Batch KG operations
- `file_watcher.py`: Real-time KG updates
- `server.py`: Query endpoints already exist

### New Components Needed

- `entity_extractor.py`: Entity extraction logic
- `relationship_discovery.py`: Relationship algorithms
- `graph_updater.py`: Incremental update logic
- `graph_query.py`: Advanced query strategies

## Key Design Decisions

1. **Async-First**: All operations use async/await for performance
2. **Batch Processing**: Optimize for throughput with batching
3. **Redis-Native**: Leverage Redis data structures directly
4. **NetworkX Integration**: Use for advanced graph algorithms
5. **Incremental Updates**: Support real-time graph evolution
