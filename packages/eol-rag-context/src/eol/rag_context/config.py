"""
Configuration for EOL RAG Context MCP Server.
"""

from typing import Optional, Dict, Any, List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict
from pathlib import Path


class RedisConfig(BaseSettings):
    """Redis connection configuration."""
    model_config = ConfigDict(env_prefix="REDIS_")
    
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: Optional[str] = Field(default=None)
    decode_responses: bool = False  # Keep False for binary vector data
    max_connections: int = Field(default=50)
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[int, int] = Field(default_factory=lambda: {
        1: 1,  # TCP_KEEPIDLE
        2: 3,  # TCP_KEEPINTVL
        3: 5,  # TCP_KEEPCNT
    })
    
    @property
    def url(self) -> str:
        """Generate Redis URL from components."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""
    model_config = ConfigDict(env_prefix="EMBEDDING_")
    
    provider: str = Field(default="sentence-transformers")
    model_name: str = Field(default="all-MiniLM-L6-v2")
    dimension: int = Field(default=384)
    batch_size: int = Field(default=32)
    normalize: bool = Field(default=True)
    
    # Provider-specific configs
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="text-embedding-3-small")
    
    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, v, info):
        """Validate embedding dimension matches model."""
        provider = info.data.get("provider", "sentence-transformers") if hasattr(info, 'data') else "sentence-transformers"
        model = info.data.get("model_name", "") if hasattr(info, 'data') else ""
        
        # Known model dimensions
        model_dims = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        
        if model in model_dims and v != model_dims[model]:
            return model_dims[model]
        return v


class IndexConfig(BaseSettings):
    """Vector index configuration."""
    model_config = ConfigDict(env_prefix="INDEX_")
    
    # Index parameters
    index_name: str = Field(default="eol_context")
    prefix: str = Field(default="doc:")
    
    # HNSW parameters
    algorithm: str = Field(default="HNSW")
    distance_metric: str = Field(default="COSINE")
    initial_cap: int = Field(default=10000)
    m: int = Field(default=16)  # Number of bi-directional links
    ef_construction: int = Field(default=200)
    ef_runtime: int = Field(default=10)
    
    # Hierarchical levels
    hierarchy_levels: int = Field(default=3)
    concept_prefix: str = Field(default="concept:")
    section_prefix: str = Field(default="section:")
    chunk_prefix: str = Field(default="chunk:")


class ChunkingConfig(BaseSettings):
    """Document chunking configuration."""
    model_config = ConfigDict(env_prefix="CHUNK_")
    
    # Chunk sizes (in tokens)
    min_chunk_size: int = Field(default=100)
    max_chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)
    
    # Semantic chunking
    use_semantic_chunking: bool = Field(default=True)
    semantic_threshold: float = Field(default=0.7)
    
    # Code chunking
    code_chunk_by_function: bool = Field(default=True)
    code_max_lines: int = Field(default=100)
    
    # Document-specific settings
    respect_document_structure: bool = Field(default=True)
    markdown_split_headers: bool = Field(default=True)


class CacheConfig(BaseSettings):
    """Semantic caching configuration."""
    model_config = ConfigDict(env_prefix="CACHE_")
    
    enabled: bool = Field(default=True)
    ttl_seconds: int = Field(default=3600)
    similarity_threshold: float = Field(default=0.97)
    max_cache_size: int = Field(default=1000)
    
    # Cache hit rate optimization
    target_hit_rate: float = Field(default=0.31)
    adaptive_threshold: bool = Field(default=True)


class ContextConfig(BaseSettings):
    """Context composition configuration."""
    model_config = ConfigDict(env_prefix="CONTEXT_")
    
    # Token limits
    max_context_tokens: int = Field(default=32000)
    reserve_tokens_for_response: int = Field(default=4000)
    
    # Retrieval parameters
    default_top_k: int = Field(default=10)
    min_relevance_score: float = Field(default=0.7)
    
    # Context strategy
    use_hierarchical_retrieval: bool = Field(default=True)
    progressive_loading: bool = Field(default=True)
    
    # Quality filters
    remove_redundancy: bool = Field(default=True)
    redundancy_threshold: float = Field(default=0.9)


class DocumentConfig(BaseSettings):
    """Document processing configuration."""
    model_config = ConfigDict(env_prefix="DOC_")
    
    # Supported file patterns
    file_patterns: List[str] = Field(
        default_factory=lambda: [
            "*.md", "*.txt", "*.json", "*.yaml", "*.yml",
            "*.py", "*.js", "*.ts", "*.jsx", "*.tsx",
            "*.java", "*.go", "*.rs", "*.cpp", "*.c",
            "*.pdf", "*.docx", "*.doc",
        ]
    )
    
    # Processing options
    extract_metadata: bool = Field(default=True)
    detect_language: bool = Field(default=True)
    parse_code_structure: bool = Field(default=True)
    
    # Size limits
    max_file_size_mb: int = Field(default=100)
    skip_binary_files: bool = Field(default=True)


class RAGConfig(BaseSettings):
    """Main RAG configuration aggregating all sub-configs."""
    model_config = ConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    # Sub-configurations
    redis: RedisConfig = Field(default_factory=RedisConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)
    
    # Server settings
    server_name: str = Field(default="eol-rag-context")
    server_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)
    
    # Storage paths
    data_dir: Path = Field(default=Path("./data"))
    index_dir: Path = Field(default=Path("./indexes"))
    
    @field_validator("data_dir", "index_dir")
    @classmethod
    def create_directories(cls, v):
        """Ensure directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @classmethod
    def from_file(cls, config_path: Path) -> "RAGConfig":
        """Load configuration from file."""
        import json
        import yaml
        
        if config_path.suffix == ".json":
            with open(config_path) as f:
                data = json.load(f)
        elif config_path.suffix in [".yaml", ".yml"]:
            with open(config_path) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        return cls(**data)