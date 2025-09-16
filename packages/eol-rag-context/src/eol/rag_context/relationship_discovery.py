"""
Relationship discovery module for finding cross-modal and semantic relationships.
Analyzes entities from different sources to identify connections.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CrossModalRelationType(Enum):
    """Types of cross-modal relationships."""

    CODE_REFERENCES_DATA = "code_references_data"
    DATA_FEEDS_CODE = "data_feeds_code"
    CODE_IMPLEMENTS_CONCEPT = "code_implements_concept"
    DATA_VALIDATES_CODE = "data_validates_code"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    PATTERN_MATCH = "pattern_match"
    API_ENDPOINT_MAPPING = "api_endpoint_mapping"
    CONFIG_BINDING = "config_binding"
    SCHEMA_IMPLEMENTATION = "schema_implementation"


@dataclass
class DiscoveredRelationship:
    """Represents a discovered relationship between entities."""

    source_entity_id: str
    target_entity_id: str
    relationship_type: CrossModalRelationType
    confidence: float
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "source": self.source_entity_id,
            "target": self.target_entity_id,
            "type": self.relationship_type.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


class RelationshipDiscovery:
    """Discovers relationships between entities from different modalities."""

    def __init__(self, embedding_manager=None, similarity_threshold: float = 0.7):
        """
        Initialize relationship discovery.

        Args:
            embedding_manager: Manager for computing semantic similarity
            similarity_threshold: Minimum similarity score for relationships
        """
        self.embedding_manager = embedding_manager
        self.similarity_threshold = similarity_threshold
        self.discovered_relationships: List[DiscoveredRelationship] = []

    async def discover_relationships(
        self,
        code_entities: List[Dict],
        data_entities: List[Dict],
        concept_entities: Optional[List[Dict]] = None,
    ) -> List[DiscoveredRelationship]:
        """
        Discover relationships between different types of entities.

        Args:
            code_entities: List of code entities
            data_entities: List of data entities
            concept_entities: Optional list of concept entities

        Returns:
            List of discovered relationships
        """
        self.discovered_relationships = []

        # Discover code-data relationships
        code_data_rels = await self._discover_code_data_relationships(code_entities, data_entities)
        self.discovered_relationships.extend(code_data_rels)

        # Discover semantic similarities
        if self.embedding_manager:
            semantic_rels = await self._discover_semantic_relationships(
                code_entities, data_entities, concept_entities
            )
            self.discovered_relationships.extend(semantic_rels)

        # Discover pattern-based relationships
        pattern_rels = self._discover_pattern_relationships(code_entities, data_entities)
        self.discovered_relationships.extend(pattern_rels)

        # Discover API mappings
        api_rels = self._discover_api_mappings(code_entities, data_entities)
        self.discovered_relationships.extend(api_rels)

        logger.info(f"Discovered {len(self.discovered_relationships)} relationships")
        return self.discovered_relationships

    async def _discover_code_data_relationships(
        self, code_entities: List[Dict], data_entities: List[Dict]
    ) -> List[DiscoveredRelationship]:
        """Discover relationships between code and data entities."""
        relationships = []

        # Build data entity lookup by name
        data_by_name = {}
        for data_entity in data_entities:
            name = data_entity.get("name", "").lower()
            if name:
                data_by_name[name] = data_entity

        # Search for data references in code
        for code_entity in code_entities:
            content = code_entity.get("content", "")
            entity_type = code_entity.get("type", "")

            # Skip non-relevant code entities
            if entity_type in ["import", "module"]:
                continue

            # Look for file references
            file_patterns = [
                r'["\']([^"\']+\.(csv|json|jsonl|xml|yaml|yml))["\']',
                r'open\s*\(\s*["\']([^"\']+)["\']',
                r'Path\s*\(\s*["\']([^"\']+)["\']',
            ]

            for pattern in file_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    filename = match[0] if isinstance(match, tuple) else match
                    filename_base = filename.split("/")[-1].split(".")[0].lower()

                    # Check if we have a matching data entity
                    if filename_base in data_by_name:
                        rel = DiscoveredRelationship(
                            source_entity_id=code_entity.get("id"),
                            target_entity_id=data_by_name[filename_base].get("id"),
                            relationship_type=CrossModalRelationType.CODE_REFERENCES_DATA,
                            confidence=0.9,
                            evidence={
                                "pattern": pattern,
                                "filename": filename,
                                "code_type": entity_type,
                            },
                        )
                        relationships.append(rel)

            # Look for DataFrame/collection references
            data_patterns = [
                r"(\w+)\.read_csv",
                r"(\w+)\.read_json",
                r"DataFrame\s*\(\s*(\w+)",
                r'load_data\s*\(\s*["\'](\w+)',
            ]

            for pattern in data_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    var_name = match.lower()
                    if var_name in data_by_name:
                        rel = DiscoveredRelationship(
                            source_entity_id=code_entity.get("id"),
                            target_entity_id=data_by_name[var_name].get("id"),
                            relationship_type=CrossModalRelationType.CODE_REFERENCES_DATA,
                            confidence=0.7,
                            evidence={"pattern": pattern, "variable": var_name},
                        )
                        relationships.append(rel)

        return relationships

    async def _discover_semantic_relationships(
        self,
        code_entities: List[Dict],
        data_entities: List[Dict],
        concept_entities: Optional[List[Dict]] = None,
    ) -> List[DiscoveredRelationship]:
        """Discover relationships based on semantic similarity."""
        relationships = []

        if not self.embedding_manager:
            return relationships

        # Combine all entities
        all_entities = []
        all_entities.extend([(e, "code") for e in code_entities])
        all_entities.extend([(e, "data") for e in data_entities])
        if concept_entities:
            all_entities.extend([(e, "concept") for e in concept_entities])

        # Generate embeddings for entities that don't have them
        entity_embeddings = []
        for entity, entity_class in all_entities:
            if "embedding" in entity:
                entity_embeddings.append((entity, entity["embedding"], entity_class))
            else:
                # Generate embedding from entity content
                text = self._get_entity_text(entity)
                if text:
                    try:
                        embedding = await self.embedding_manager.get_embedding(text)
                        entity_embeddings.append((entity, embedding, entity_class))
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding: {e}")

        # Find similar entities across different modalities
        for i, (entity1, emb1, class1) in enumerate(entity_embeddings):
            for j, (entity2, emb2, class2) in enumerate(entity_embeddings):
                if i >= j or class1 == class2:
                    continue  # Skip same class or already compared

                # Compute cosine similarity
                similarity = self._cosine_similarity(emb1, emb2)

                if similarity >= self.similarity_threshold:
                    rel = DiscoveredRelationship(
                        source_entity_id=entity1.get("id"),
                        target_entity_id=entity2.get("id"),
                        relationship_type=CrossModalRelationType.SEMANTIC_SIMILARITY,
                        confidence=float(similarity),
                        evidence={
                            "similarity_score": float(similarity),
                            "source_type": class1,
                            "target_type": class2,
                        },
                    )
                    relationships.append(rel)

        return relationships

    def _discover_pattern_relationships(
        self, code_entities: List[Dict], data_entities: List[Dict]
    ) -> List[DiscoveredRelationship]:
        """Discover relationships based on naming patterns."""
        relationships = []

        # Common naming patterns
        patterns = [
            (r"get_(\w+)", r"\1"),  # get_users -> users
            (r"fetch_(\w+)", r"\1"),  # fetch_orders -> orders
            (r"load_(\w+)", r"\1"),  # load_config -> config
            (r"save_(\w+)", r"\1"),  # save_results -> results
            (r"(\w+)_handler", r"\1"),  # user_handler -> user
            (r"(\w+)_processor", r"\1"),  # data_processor -> data
            (r"(\w+)_model", r"\1"),  # user_model -> user
            (r"(\w+)_schema", r"\1"),  # product_schema -> product
        ]

        # Build pattern index for code entities
        code_patterns = defaultdict(list)
        for code_entity in code_entities:
            name = code_entity.get("name", "")
            for pattern, replacement in patterns:
                match = re.match(pattern, name, re.IGNORECASE)
                if match:
                    base_name = re.sub(pattern, replacement, name, flags=re.IGNORECASE)
                    code_patterns[base_name.lower()].append(code_entity)

        # Match with data entities
        for data_entity in data_entities:
            data_name = data_entity.get("name", "").lower()

            # Direct pattern match
            if data_name in code_patterns:
                for code_entity in code_patterns[data_name]:
                    rel = DiscoveredRelationship(
                        source_entity_id=code_entity.get("id"),
                        target_entity_id=data_entity.get("id"),
                        relationship_type=CrossModalRelationType.PATTERN_MATCH,
                        confidence=0.8,
                        evidence={
                            "pattern": "naming_convention",
                            "code_name": code_entity.get("name"),
                            "data_name": data_entity.get("name"),
                        },
                    )
                    relationships.append(rel)

            # Check for plural/singular variations
            variations = [
                data_name + "s",  # user -> users
                data_name[:-1] if data_name.endswith("s") else None,  # users -> user
            ]

            for variation in variations:
                if variation and variation in code_patterns:
                    for code_entity in code_patterns[variation]:
                        rel = DiscoveredRelationship(
                            source_entity_id=code_entity.get("id"),
                            target_entity_id=data_entity.get("id"),
                            relationship_type=CrossModalRelationType.PATTERN_MATCH,
                            confidence=0.7,
                            evidence={
                                "pattern": "plural_singular_variation",
                                "code_name": code_entity.get("name"),
                                "data_name": data_entity.get("name"),
                            },
                        )
                        relationships.append(rel)

        return relationships

    def _discover_api_mappings(
        self, code_entities: List[Dict], data_entities: List[Dict]
    ) -> List[DiscoveredRelationship]:
        """Discover API endpoint mappings between code and data."""
        relationships = []

        # Find API endpoints in code
        api_endpoints = []
        endpoint_patterns = [
            r'@app\.route\s*\(\s*["\']([^"\']+)',  # Flask
            r'@router\.(get|post|put|delete)\s*\(\s*["\']([^"\']+)',  # FastAPI
            r'path\s*\(\s*["\']([^"\']+)',  # Django
            r'app\.(get|post|put|delete)\s*\(\s*["\']([^"\']+)',  # Express-like
        ]

        for code_entity in code_entities:
            content = code_entity.get("content", "")
            for pattern in endpoint_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    endpoint = match[-1] if isinstance(match, tuple) else match
                    api_endpoints.append((code_entity, endpoint))

        # Match endpoints with data entities
        for code_entity, endpoint in api_endpoints:
            # Extract resource name from endpoint
            # e.g., /api/users/{id} -> users
            resource_match = re.search(r"/([a-zA-Z_]+)", endpoint)
            if resource_match:
                resource_name = resource_match.group(1).lower()

                for data_entity in data_entities:
                    data_name = data_entity.get("name", "").lower()

                    # Check if resource name matches data entity
                    if resource_name == data_name or resource_name == data_name + "s":
                        rel = DiscoveredRelationship(
                            source_entity_id=code_entity.get("id"),
                            target_entity_id=data_entity.get("id"),
                            relationship_type=CrossModalRelationType.API_ENDPOINT_MAPPING,
                            confidence=0.85,
                            evidence={
                                "endpoint": endpoint,
                                "resource": resource_name,
                                "method": code_entity.get("name", ""),
                            },
                        )
                        relationships.append(rel)

        return relationships

    def discover_config_bindings(
        self, code_entities: List[Dict], config_data: Dict
    ) -> List[DiscoveredRelationship]:
        """Discover configuration bindings between code and config files."""
        relationships = []

        # Extract config keys
        config_keys = self._extract_config_keys(config_data)

        for code_entity in code_entities:
            content = code_entity.get("content", "")

            # Look for config access patterns
            config_patterns = [
                r'config\[[\'"]([\w.]+)[\'"]',  # config["key"]
                r'config\.get\s*\(\s*[\'"]([\w.]+)',  # config.get("key")
                r'os\.environ\[[\'"]([\w_]+)',  # os.environ["KEY"]
                r'getenv\s*\(\s*[\'"]([\w_]+)',  # getenv("KEY")
            ]

            for pattern in config_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    config_key = match.lower()

                    if config_key in config_keys:
                        rel = DiscoveredRelationship(
                            source_entity_id=code_entity.get("id"),
                            target_entity_id=f"config_{config_key}",
                            relationship_type=CrossModalRelationType.CONFIG_BINDING,
                            confidence=0.9,
                            evidence={
                                "config_key": config_key,
                                "pattern": pattern,
                                "code_location": code_entity.get("name"),
                            },
                        )
                        relationships.append(rel)

        return relationships

    def _get_entity_text(self, entity: Dict) -> str:
        """Extract text representation from an entity for embedding."""
        parts = []

        # Add name
        if "name" in entity:
            parts.append(f"Name: {entity['name']}")

        # Add type
        if "type" in entity:
            parts.append(f"Type: {entity['type']}")

        # Add docstring or description
        if "docstring" in entity:
            parts.append(f"Description: {entity['docstring']}")
        elif "description" in entity:
            parts.append(f"Description: {entity['description']}")

        # Add limited content
        if "content" in entity:
            content = str(entity["content"])[:300]
            parts.append(f"Content: {content}")

        return " ".join(parts)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)
        except Exception as e:
            logger.warning(f"Error computing cosine similarity: {e}")
            return 0.0

    def _extract_config_keys(self, config_data: Dict, prefix: str = "") -> set:
        """Recursively extract all configuration keys."""
        keys = set()

        for key, value in config_data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.add(full_key.lower())

            # Recursively process nested dicts
            if isinstance(value, dict):
                nested_keys = self._extract_config_keys(value, full_key)
                keys.update(nested_keys)

        return keys

    def merge_duplicate_relationships(
        self, min_confidence: float = 0.5
    ) -> List[DiscoveredRelationship]:
        """Merge duplicate relationships and filter by confidence."""
        # Group relationships by source-target pair
        relationship_groups = defaultdict(list)

        for rel in self.discovered_relationships:
            if rel.confidence >= min_confidence:
                key = (rel.source_entity_id, rel.target_entity_id)
                relationship_groups[key].append(rel)

        # Merge groups, keeping highest confidence
        merged = []
        for (source, target), rels in relationship_groups.items():
            if len(rels) == 1:
                merged.append(rels[0])
            else:
                # Merge multiple relationships
                best_rel = max(rels, key=lambda r: r.confidence)

                # Combine evidence from all relationships
                combined_evidence = {}
                for rel in rels:
                    combined_evidence[rel.relationship_type.value] = rel.evidence

                best_rel.evidence = combined_evidence
                merged.append(best_rel)

        return merged
