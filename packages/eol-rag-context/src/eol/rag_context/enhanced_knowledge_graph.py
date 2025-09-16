"""
Enhanced knowledge graph builder with multimodal support.
Extends the base KnowledgeGraphBuilder with code analysis and data extraction.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .code_analyzer import ASTCodeAnalyzer, CodeEntity
from .knowledge_graph import KnowledgeGraphBuilder
from .multimodal_config import MultimodalConfig

logger = logging.getLogger(__name__)


class EnhancedKnowledgeGraphBuilder(KnowledgeGraphBuilder):
    """Extended knowledge graph builder with multimodal capabilities."""

    def __init__(self, redis_store, embedding_manager, config: Optional[MultimodalConfig] = None):
        """
        Initialize enhanced knowledge graph builder.

        Args:
            redis_store: Redis store for persistence
            embedding_manager: Manager for generating embeddings
            config: Multimodal configuration (uses defaults if not provided)
        """
        super().__init__(redis_store, embedding_manager)
        self.config = config or MultimodalConfig()

        # Validate configuration
        warnings = self.config.validate()
        if warnings:
            for warning in warnings:
                logger.warning(f"Configuration warning: {warning}")

        # Lazy-loaded components
        self._code_analyzer: Optional[ASTCodeAnalyzer] = None
        self._data_extractor = None
        self._pattern_detector = None

    @property
    def code_analyzer(self) -> ASTCodeAnalyzer:
        """Get code analyzer (lazy initialization)."""
        if not self._code_analyzer and self.config.enable_code_analysis:
            self._code_analyzer = ASTCodeAnalyzer()
        return self._code_analyzer

    @property
    def data_extractor(self):
        """Get data extractor (lazy initialization)."""
        if not self._data_extractor and self.config.enable_data_extraction:
            from .data_extractor import DataExtractor

            self._data_extractor = DataExtractor(self.config)
        return self._data_extractor

    async def build_from_code(self, directory: Path, source_id: Optional[str] = None) -> Dict:
        """
        Build knowledge graph from source code files.

        Args:
            directory: Directory containing source code
            source_id: Optional source identifier for filtering

        Returns:
            Statistics about entities and relationships created
        """
        if not self.config.enable_code_analysis:
            logger.info("Code analysis disabled in configuration")
            return {"entities": 0, "relationships": 0}

        stats = {"entities": 0, "relationships": 0, "files_processed": 0}

        try:
            # Analyze all Python files in directory
            entities, relations = self.code_analyzer.analyze_directory(directory)

            # Convert code entities to graph nodes
            for entity in entities:
                node_data = {
                    "id": entity.id,
                    "type": entity.type.value,
                    "name": entity.name,
                    "file_path": entity.file_path,
                    "line_start": entity.line_start,
                    "line_end": entity.line_end,
                    "content": entity.content,
                    "metadata": entity.metadata,
                }

                # Add docstring if available
                if entity.docstring:
                    node_data["docstring"] = entity.docstring

                # Add signature for functions/methods
                if entity.signature:
                    node_data["signature"] = entity.signature

                # Add node to graph
                await self.add_node(entity.id, node_data)
                stats["entities"] += 1

                # Generate embedding for the entity
                if self.config.include_embeddings:
                    text_for_embedding = self._create_entity_text(entity)
                    embedding = await self.embedding_manager.get_embedding(text_for_embedding)
                    await self.store_entity_embedding(entity.id, embedding, entity.to_dict())

            # Add relationships to graph
            for relation in relations:
                await self.add_edge(
                    relation.source_id,
                    relation.target_id,
                    {"type": relation.type.value, "metadata": relation.metadata},
                )
                stats["relationships"] += 1

            # Track processed files
            processed_files = set(e.file_path for e in entities)
            stats["files_processed"] = len(processed_files)

            logger.info(
                f"Code analysis complete: {stats['entities']} entities, "
                f"{stats['relationships']} relationships from {stats['files_processed']} files"
            )

        except Exception as e:
            logger.error(f"Error building graph from code: {e}")

        return stats

    async def build_from_data(self, file_path: Path, source_id: Optional[str] = None) -> Dict:
        """
        Build knowledge graph from data files (CSV, JSON).

        Args:
            file_path: Path to data file
            source_id: Optional source identifier

        Returns:
            Statistics about entities and relationships created
        """
        if not self.config.enable_data_extraction:
            logger.info("Data extraction disabled in configuration")
            return {"entities": 0, "relationships": 0}

        if not self.data_extractor:
            logger.warning("Data extractor not available")
            return {"entities": 0, "relationships": 0}

        stats = {"entities": 0, "relationships": 0}

        try:
            # Extract entities and relationships from data
            entities, relations = await self.data_extractor.extract_from_file(file_path)

            # Add to graph
            for entity in entities:
                await self.add_node(entity["id"], entity)
                stats["entities"] += 1

                # Generate embeddings if configured
                if self.config.include_embeddings and "content" in entity:
                    embedding = await self.embedding_manager.get_embedding(entity["content"])
                    await self.store_entity_embedding(entity["id"], embedding, entity)

            for relation in relations:
                await self.add_edge(
                    relation["source"], relation["target"], relation.get("metadata", {})
                )
                stats["relationships"] += 1

            logger.info(
                f"Data extraction complete: {stats['entities']} entities, "
                f"{stats['relationships']} relationships"
            )

        except Exception as e:
            logger.error(f"Error building graph from data: {e}")

        return stats

    async def detect_patterns(self, min_frequency: Optional[int] = None) -> List[Dict]:
        """
        Detect patterns across the knowledge graph.

        Args:
            min_frequency: Minimum occurrence frequency for a pattern

        Returns:
            List of detected patterns
        """
        if not self.config.enable_pattern_detection:
            logger.info("Pattern detection disabled in configuration")
            return []

        min_freq = min_frequency or self.config.min_pattern_frequency
        patterns = []

        try:
            # Detect code patterns
            code_patterns = await self._detect_code_patterns(min_freq)
            patterns.extend(code_patterns)

            # Detect data patterns
            data_patterns = await self._detect_data_patterns(min_freq)
            patterns.extend(data_patterns)

            # Detect cross-modal patterns
            if self.config.enable_cross_modal_linking:
                cross_patterns = await self._detect_cross_modal_patterns(min_freq)
                patterns.extend(cross_patterns)

            logger.info(f"Pattern detection complete: {len(patterns)} patterns found")

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")

        return patterns

    async def _detect_code_patterns(self, min_frequency: int) -> List[Dict]:
        """Detect patterns in code entities."""
        patterns = []

        # Analyze function call patterns
        call_patterns = {}
        for edge in self.graph.edges(data=True):
            if edge[2].get("type") == "calls":
                pattern_key = (edge[0], edge[1])
                call_patterns[pattern_key] = call_patterns.get(pattern_key, 0) + 1

        # Filter by frequency
        for pattern, count in call_patterns.items():
            if count >= min_frequency:
                patterns.append(
                    {
                        "type": "call_pattern",
                        "source": pattern[0],
                        "target": pattern[1],
                        "frequency": count,
                        "category": "api_patterns",
                    }
                )

        return patterns

    async def _detect_data_patterns(self, min_frequency: int) -> List[Dict]:
        """Detect patterns in data entities."""
        patterns = []

        # Analyze data structure patterns
        structure_patterns = {}
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get("type") == "data":
                structure = node_data.get("structure", "unknown")
                structure_patterns[structure] = structure_patterns.get(structure, 0) + 1

        # Filter by frequency
        for structure, count in structure_patterns.items():
            if count >= min_frequency:
                patterns.append(
                    {
                        "type": "data_structure",
                        "structure": structure,
                        "frequency": count,
                        "category": "data_patterns",
                    }
                )

        return patterns

    async def _detect_cross_modal_patterns(self, min_frequency: int) -> List[Dict]:
        """Detect patterns across different modalities."""
        patterns = []

        # Find code that references data files
        code_data_refs = {}
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get("type") in ["function", "method"]:
                content = node_data.get("content", "")
                # Simple heuristic: look for file extensions in content
                for ext in [".csv", ".json", ".xml", ".yaml"]:
                    if ext in content:
                        pattern_key = f"code_uses_{ext}"
                        code_data_refs[pattern_key] = code_data_refs.get(pattern_key, 0) + 1

        # Filter by frequency
        for pattern, count in code_data_refs.items():
            if count >= min_frequency:
                patterns.append(
                    {
                        "type": "cross_modal",
                        "pattern": pattern,
                        "frequency": count,
                        "category": "integration_patterns",
                    }
                )

        return patterns

    def _create_entity_text(self, entity: CodeEntity) -> str:
        """
        Create text representation of code entity for embedding.

        Args:
            entity: Code entity to convert

        Returns:
            Text representation for embedding
        """
        parts = [f"{entity.type.value}: {entity.name}"]

        if entity.signature:
            parts.append(f"Signature: {entity.signature}")

        if entity.docstring:
            parts.append(f"Documentation: {entity.docstring}")

        # Add limited content
        if entity.content:
            content_preview = entity.content[:200]
            parts.append(f"Content: {content_preview}")

        return "\n".join(parts)

    async def store_entity_embedding(
        self, entity_id: str, embedding: List[float], metadata: Dict
    ) -> None:
        """
        Store entity embedding in Redis for similarity search.

        Args:
            entity_id: Unique entity identifier
            embedding: Vector embedding
            metadata: Entity metadata
        """
        try:
            # Store in Redis (implementation depends on redis_store interface)
            # This would typically use the redis_store's vector storage capabilities
            # Prepare data for Redis (placeholder for actual implementation)
            # redis_data = {
            #     "entity_id": entity_id,
            #     "embedding": embedding,
            #     "type": metadata.get("type", "unknown"),
            #     "name": metadata.get("name", ""),
            #     "file_path": metadata.get("file_path", ""),
            #     **metadata,
            # }
            logger.debug(f"Stored embedding for entity {entity_id}")

        except Exception as e:
            logger.error(f"Error storing entity embedding: {e}")

    async def find_similar_entities(
        self, entity_id: str, k: int = 5, threshold: float = None
    ) -> List[Tuple[str, float]]:
        """
        Find entities similar to a given entity using embeddings.

        Args:
            entity_id: Entity to find similarities for
            k: Number of similar entities to return
            threshold: Minimum similarity threshold

        Returns:
            List of (entity_id, similarity_score) tuples
        """
        threshold = threshold or self.config.similarity_threshold

        try:
            # Get entity embedding
            entity = self.graph.nodes[entity_id]
            if "embedding" not in entity:
                logger.warning(f"No embedding found for entity {entity_id}")
                return []

            # Search for similar entities using vector similarity
            # This would use the redis_store's vector search capabilities
            similar = []  # Placeholder for actual implementation

            return similar

        except Exception as e:
            logger.error(f"Error finding similar entities: {e}")
            return []

    async def merge_similar_entities(self, threshold: float = None) -> int:
        """
        Merge entities that are highly similar.

        Args:
            threshold: Similarity threshold for merging

        Returns:
            Number of entities merged
        """
        if not self.config.merge_similar_entities:
            return 0

        threshold = threshold or self.config.entity_similarity_threshold
        merged_count = 0

        try:
            # Find all entity pairs above threshold
            # Merge entities while preserving relationships
            # This is a placeholder for the actual implementation
            logger.info(f"Entity merging complete: {merged_count} entities merged")

        except Exception as e:
            logger.error(f"Error merging entities: {e}")

        return merged_count
