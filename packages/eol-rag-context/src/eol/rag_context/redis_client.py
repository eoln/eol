"""
Redis 8 client with vector capabilities for EOL RAG Context.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import logging

try:
    from redis import Redis
    from redis.asyncio import Redis as AsyncRedis
    from redis.commands.search.field import TextField, VectorField, NumericField, TagField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
except ImportError:
    # Fallback for testing without redis-py[search]
    from unittest.mock import MagicMock
    Redis = MagicMock
    AsyncRedis = MagicMock
    TextField = MagicMock
    VectorField = MagicMock
    NumericField = MagicMock
    TagField = MagicMock
    IndexDefinition = MagicMock
    IndexType = MagicMock
    Query = MagicMock

from .config import RedisConfig, IndexConfig

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Document with vector embedding."""
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    hierarchy_level: int = 1  # 1=concept, 2=section, 3=chunk
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)


class RedisVectorStore:
    """Redis 8 vector store with hierarchical indexing."""
    
    def __init__(self, redis_config: RedisConfig, index_config: IndexConfig):
        self.redis_config = redis_config
        self.index_config = index_config
        self.redis: Optional[Redis] = None
        self.async_redis: Optional[AsyncRedis] = None
        
    def connect(self) -> None:
        """Establish Redis connection."""
        self.redis = Redis(
            host=self.redis_config.host,
            port=self.redis_config.port,
            db=self.redis_config.db,
            password=self.redis_config.password,
            decode_responses=self.redis_config.decode_responses,
            max_connections=self.redis_config.max_connections,
            socket_keepalive=self.redis_config.socket_keepalive,
            socket_keepalive_options=self.redis_config.socket_keepalive_options,
        )
        
        # Test connection
        try:
            self.redis.ping()
            logger.info(f"Connected to Redis at {self.redis_config.host}:{self.redis_config.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def connect_async(self) -> None:
        """Establish async Redis connection."""
        self.async_redis = await AsyncRedis(
            host=self.redis_config.host,
            port=self.redis_config.port,
            db=self.redis_config.db,
            password=self.redis_config.password,
            decode_responses=self.redis_config.decode_responses,
            max_connections=self.redis_config.max_connections,
            socket_keepalive=self.redis_config.socket_keepalive,
            socket_keepalive_options=self.redis_config.socket_keepalive_options,
        )
        
        # Test connection
        try:
            await self.async_redis.ping()
            logger.info(f"Async connected to Redis at {self.redis_config.host}:{self.redis_config.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def create_hierarchical_indexes(self, embedding_dim: int) -> None:
        """Create indexes for each hierarchy level."""
        
        # Define schemas for each level
        schemas = {
            "concept": {
                "prefix": self.index_config.concept_prefix,
                "fields": [
                    TextField("content", weight=1.0),
                    VectorField(
                        "embedding",
                        self.index_config.algorithm,
                        {
                            "TYPE": "FLOAT32",
                            "DIM": embedding_dim,
                            "DISTANCE_METRIC": self.index_config.distance_metric,
                            "INITIAL_CAP": self.index_config.initial_cap,
                            "M": self.index_config.m,
                            "EF_CONSTRUCTION": self.index_config.ef_construction,
                        }
                    ),
                    TagField("children"),
                    NumericField("created_at"),
                    TextField("metadata"),
                ]
            },
            "section": {
                "prefix": self.index_config.section_prefix,
                "fields": [
                    TextField("content", weight=0.8),
                    VectorField(
                        "embedding",
                        self.index_config.algorithm,
                        {
                            "TYPE": "FLOAT32",
                            "DIM": embedding_dim,
                            "DISTANCE_METRIC": self.index_config.distance_metric,
                            "INITIAL_CAP": self.index_config.initial_cap * 10,
                            "M": self.index_config.m + 8,
                            "EF_CONSTRUCTION": self.index_config.ef_construction + 100,
                        }
                    ),
                    TagField("parent"),
                    TagField("children"),
                    NumericField("created_at"),
                    TextField("metadata"),
                ]
            },
            "chunk": {
                "prefix": self.index_config.chunk_prefix,
                "fields": [
                    TextField("content", weight=0.6),
                    VectorField(
                        "embedding",
                        "FLAT",  # Use FLAT for chunk level (exact search)
                        {
                            "TYPE": "FLOAT32",
                            "DIM": embedding_dim,
                            "DISTANCE_METRIC": self.index_config.distance_metric,
                            "INITIAL_CAP": self.index_config.initial_cap * 100,
                        }
                    ),
                    TagField("parent"),
                    NumericField("position"),
                    NumericField("created_at"),
                    TextField("metadata"),
                    TagField("doc_type"),  # markdown, code, pdf, etc.
                    TagField("language"),  # for code files
                ]
            }
        }
        
        # Create indexes
        for level_name, schema in schemas.items():
            index_name = f"{self.index_config.index_name}_{level_name}"
            
            try:
                # Check if index exists
                self.redis.ft(index_name).info()
                logger.info(f"Index {index_name} already exists")
            except:
                # Create new index
                definition = IndexDefinition(
                    prefix=[schema["prefix"]],
                    index_type=IndexType.HASH
                )
                
                self.redis.ft(index_name).create_index(
                    fields=schema["fields"],
                    definition=definition
                )
                logger.info(f"Created index {index_name}")
    
    async def store_document(self, doc: VectorDocument) -> None:
        """Store document with hierarchical metadata."""
        if not self.async_redis:
            await self.connect_async()
        
        # Determine prefix based on hierarchy level
        prefix_map = {
            1: self.index_config.concept_prefix,
            2: self.index_config.section_prefix,
            3: self.index_config.chunk_prefix,
        }
        prefix = prefix_map.get(doc.hierarchy_level, self.index_config.chunk_prefix)
        
        # Prepare data for storage
        key = f"{prefix}{doc.id}"
        data = {
            "content": doc.content,
            "embedding": doc.embedding.astype(np.float32).tobytes(),
            "metadata": json.dumps(doc.metadata),
            "created_at": doc.metadata.get("created_at", 0),
        }
        
        # Add hierarchy-specific fields
        if doc.parent_id:
            data["parent"] = doc.parent_id
        if doc.children_ids:
            data["children"] = ",".join(doc.children_ids)
        if doc.hierarchy_level == 3:  # Chunk level
            data["position"] = doc.metadata.get("position", 0)
            data["doc_type"] = doc.metadata.get("doc_type", "unknown")
            data["language"] = doc.metadata.get("language", "unknown")
        
        # Store in Redis
        await self.async_redis.hset(key, mapping=data)
        logger.debug(f"Stored document {key}")
    
    async def vector_search(
        self,
        query_embedding: np.ndarray,
        hierarchy_level: int = 3,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform vector similarity search at specified hierarchy level.
        
        Returns list of (id, score, data) tuples.
        """
        if not self.async_redis:
            await self.connect_async()
        
        # Select appropriate index
        level_map = {1: "concept", 2: "section", 3: "chunk"}
        index_name = f"{self.index_config.index_name}_{level_map[hierarchy_level]}"
        
        # Build query
        query_vector = query_embedding.astype(np.float32).tobytes()
        
        # Base KNN query
        query_str = f"*=>[KNN {k} @embedding $vec AS score]"
        
        # Add filters if provided
        if filters:
            filter_clauses = []
            for field, value in filters.items():
                if isinstance(value, str):
                    filter_clauses.append(f"@{field}:{{{value}}}")
                elif isinstance(value, (int, float)):
                    filter_clauses.append(f"@{field}:[{value} {value}]")
            
            if filter_clauses:
                query_str = f"({' '.join(filter_clauses)}) {query_str}"
        
        # Create query
        query = (
            Query(query_str)
            .return_fields("content", "metadata", "score", "parent", "children")
            .sort_by("score", asc=True)
            .dialect(2)
        )
        
        # Execute search
        results = await self.async_redis.ft(index_name).search(
            query,
            query_params={"vec": query_vector}
        )
        
        # Parse results
        output = []
        for doc in results.docs:
            doc_id = doc.id.split(":")[-1]
            score = float(doc.score) if hasattr(doc, 'score') else 0.0
            
            data = {
                "content": doc.content if hasattr(doc, 'content') else "",
                "metadata": json.loads(doc.metadata) if hasattr(doc, 'metadata') else {},
                "parent": doc.parent if hasattr(doc, 'parent') else None,
                "children": doc.children.split(",") if hasattr(doc, 'children') else [],
            }
            
            output.append((doc_id, score, data))
        
        return output
    
    async def hierarchical_search(
        self,
        query_embedding: np.ndarray,
        max_chunks: int = 10,
        strategy: str = "adaptive"
    ) -> List[Dict[str, Any]]:
        """
        Perform hierarchical search starting from concepts down to chunks.
        """
        results = []
        
        # Step 1: Find relevant concepts
        concepts = await self.vector_search(
            query_embedding,
            hierarchy_level=1,
            k=3
        )
        
        if not concepts:
            # Fallback to direct chunk search
            chunks = await self.vector_search(
                query_embedding,
                hierarchy_level=3,
                k=max_chunks
            )
            return [{"id": c[0], "score": c[1], **c[2]} for c in chunks]
        
        # Step 2: Find sections within concepts
        sections = []
        for concept_id, concept_score, concept_data in concepts:
            concept_sections = await self.vector_search(
                query_embedding,
                hierarchy_level=2,
                k=5,
                filters={"parent": concept_id}
            )
            
            for sec_id, sec_score, sec_data in concept_sections:
                sections.append({
                    "id": sec_id,
                    "score": sec_score * 0.8 + concept_score * 0.2,  # Weighted score
                    "data": sec_data,
                    "concept_id": concept_id
                })
        
        # Sort sections by combined score
        sections.sort(key=lambda x: x["score"], reverse=True)
        sections = sections[:10]  # Keep top sections
        
        # Step 3: Get specific chunks if needed
        if strategy == "detailed" or len(sections) < 3:
            for section in sections[:5]:
                section_chunks = await self.vector_search(
                    query_embedding,
                    hierarchy_level=3,
                    k=3,
                    filters={"parent": section["id"]}
                )
                
                for chunk_id, chunk_score, chunk_data in section_chunks:
                    results.append({
                        "id": chunk_id,
                        "score": chunk_score * 0.6 + section["score"] * 0.4,
                        "content": chunk_data["content"],
                        "metadata": chunk_data["metadata"],
                        "hierarchy": {
                            "concept": concept_id,
                            "section": section["id"],
                            "chunk": chunk_id
                        }
                    })
        else:
            # Return section-level content
            results = sections
        
        # Sort by final score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_chunks]
    
    async def get_document_tree(self, doc_id: str) -> Dict[str, Any]:
        """Get full document tree from any node."""
        if not self.async_redis:
            await self.connect_async()
        
        # Try each prefix to find the document
        for prefix in [self.index_config.concept_prefix, 
                      self.index_config.section_prefix,
                      self.index_config.chunk_prefix]:
            key = f"{prefix}{doc_id}"
            data = await self.async_redis.hgetall(key)
            
            if data:
                result = {
                    "id": doc_id,
                    "content": data.get(b"content", b"").decode(),
                    "metadata": json.loads(data.get(b"metadata", b"{}").decode()),
                }
                
                # Add parent if exists
                if b"parent" in data:
                    result["parent"] = data[b"parent"].decode()
                    parent_tree = await self.get_document_tree(result["parent"])
                    result["parent_data"] = parent_tree
                
                # Add children if exist
                if b"children" in data:
                    children_ids = data[b"children"].decode().split(",")
                    result["children"] = []
                    for child_id in children_ids:
                        if child_id:
                            child_tree = await self.get_document_tree(child_id)
                            result["children"].append(child_tree)
                
                return result
        
        return {"id": doc_id, "error": "Not found"}
    
    async def close(self) -> None:
        """Close Redis connections."""
        if self.redis:
            self.redis.close()
        if self.async_redis:
            await self.async_redis.close()