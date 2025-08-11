"""
Configuration classes for EOL RAG Context MCP Server.

This module provides comprehensive configuration management for all aspects
of the RAG system including Redis connections, embedding models, vector indexes,
document processing, semantic caching, and context management.

The configuration system uses Pydantic Settings for validation, environment
variable support, and type safety. Each component has its own configuration
class that can be customized independently.

Example:
    Basic usage with default settings:

    >>> from eol.rag_context.config import RAGConfig
    >>> config = RAGConfig()
    >>> print(f"Redis host: {config.redis.host}")
    Redis host: localhost

    Custom configuration from file:

    >>> config = RAGConfig.from_file(Path("config.yaml"))
    >>> config.embedding.model_name = "all-mpnet-base-v2"
    >>> config.redis.host = "redis.example.com"
"""

from typing import Optional, Dict, Any, List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict
from pathlib import Path


class RedisConfig(BaseSettings):
    """Redis connection and pool configuration for vector storage.

    Configures Redis connection parameters including host, port, authentication,
    and connection pooling options. Supports both standalone and clustered Redis
    deployments with optimized settings for vector operations.

    Environment variables can be prefixed with REDIS_ (e.g., REDIS_HOST=localhost).

    Attributes:
        host: Redis server hostname or IP address.
        port: Redis server port number.
        db: Redis database number to use.
        password: Optional Redis password for authentication.
        decode_responses: Whether to decode Redis responses to strings.
            Should be False for vector data to preserve binary format.
        max_connections: Maximum number of connections in the pool.
        socket_keepalive: Enable TCP socket keepalive.
        socket_keepalive_options: TCP keepalive parameters for connection health.

    Example:
        >>> redis_config = RedisConfig(
        ...     host="redis.example.com",
        ...     port=6380,
        ...     password="secret"
        ... )
        >>> print(redis_config.url)
        redis://:secret@redis.example.com:6380/0
    """

    model_config = ConfigDict(env_prefix="REDIS_")

    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: Optional[str] = Field(default=None)
    decode_responses: bool = False  # Keep False for binary vector data
    max_connections: int = Field(default=50)
    socket_keepalive: bool = True
    socket_keepalive_options: Dict[int, int] = Field(
        default_factory=lambda: {
            1: 1,  # TCP_KEEPIDLE
            2: 3,  # TCP_KEEPINTVL
            3: 5,  # TCP_KEEPCNT
        }
    )

    @property
    def url(self) -> str:
        """Generate Redis connection URL from configuration components.

        Constructs a Redis URL in the format redis://[password@]host:port/db
        suitable for use with Redis clients and connection libraries.

        Returns:
            Complete Redis connection URL string.

        Example:
            >>> config = RedisConfig(host="localhost", port=6379, password="secret")
            >>> config.url
            'redis://:secret@localhost:6379/0'
        """
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration and provider settings.

    Configures embedding model providers, model selection, and generation parameters.
    Supports multiple providers including Sentence Transformers (local), OpenAI,
    and other cloud-based embedding services.

    Environment variables can be prefixed with EMBEDDING_ (e.g., EMBEDDING_PROVIDER=openai).

    Attributes:
        provider: Embedding provider name (sentence-transformers, openai).
        model_name: Specific model to use for embedding generation.
        dimension: Vector dimension of the embedding model.
        batch_size: Number of texts to process in each batch.
        normalize: Whether to normalize embedding vectors to unit length.
        openai_api_key: API key for OpenAI embedding services.
        openai_model: OpenAI model name for embedding generation.

    Example:
        Local embeddings:

        >>> config = EmbeddingConfig(
        ...     provider="sentence-transformers",
        ...     model_name="all-mpnet-base-v2"
        ... )
        >>> print(f"Dimensions: {config.dimension}")
        Dimensions: 768

        OpenAI embeddings:

        >>> config = EmbeddingConfig(
        ...     provider="openai",
        ...     openai_api_key="sk-...",
        ...     openai_model="text-embedding-3-large"
        ... )
    """

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
        """Validate and auto-correct embedding dimension based on model.

        Automatically sets the correct dimension for known models to prevent
        configuration errors. Falls back to provided dimension for unknown models.

        Args:
            v: The provided dimension value.
            info: Pydantic validation info containing other field values.

        Returns:
            Corrected dimension value for the specified model.

        Note:
            Known model dimensions:
            - all-MiniLM-L6-v2: 384
            - all-mpnet-base-v2: 768
            - text-embedding-3-small: 1536
            - text-embedding-3-large: 3072
        """
        provider = (
            info.data.get("provider", "sentence-transformers")
            if hasattr(info, "data")
            else "sentence-transformers"
        )
        model = info.data.get("model_name", "") if hasattr(info, "data") else ""

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
    """Vector index configuration for Redis Search with HNSW algorithm.

    Configures vector index parameters for optimal search performance including
    HNSW (Hierarchical Navigable Small World) algorithm settings, distance metrics,
    and hierarchical organization for concepts, sections, and chunks.

    Environment variables can be prefixed with INDEX_ (e.g., INDEX_ALGORITHM=HNSW).

    Attributes:
        index_name: Base name for the vector index.
        prefix: Key prefix for all indexed documents.
        algorithm: Vector indexing algorithm (HNSW recommended).
        distance_metric: Distance metric for similarity calculation (COSINE, L2, IP).
        initial_cap: Initial capacity hint for index size.
        m: HNSW parameter - number of bi-directional links per node.
        ef_construction: HNSW parameter - size of candidate set during construction.
        ef_runtime: HNSW parameter - size of candidate set during search.
        hierarchy_levels: Number of hierarchical levels (typically 3).
        concept_prefix: Key prefix for concept-level documents.
        section_prefix: Key prefix for section-level documents.
        chunk_prefix: Key prefix for chunk-level documents.

    Example:
        >>> index_config = IndexConfig(
        ...     index_name="project_context",
        ...     m=32,  # Higher connectivity for better recall
        ...     ef_construction=400  # Higher quality construction
        ... )
        >>> print(f"Algorithm: {index_config.algorithm}")
        Algorithm: HNSW
    """

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
    """Document chunking configuration for optimal content segmentation.

    Configures how documents are split into chunks for embedding and indexing.
    Supports both token-based and semantic chunking strategies with special
    handling for different document types (code, markdown, plain text).

    Environment variables can be prefixed with CHUNK_ (e.g., CHUNK_MAX_CHUNK_SIZE=1024).

    Attributes:
        min_chunk_size: Minimum chunk size in tokens to avoid tiny fragments.
        max_chunk_size: Maximum chunk size in tokens to fit embedding context.
        chunk_overlap: Number of overlapping tokens between adjacent chunks.
        use_semantic_chunking: Whether to use semantic similarity for chunk boundaries.
        semantic_threshold: Similarity threshold for semantic chunk boundaries.
        code_chunk_by_function: Whether to chunk code files by function/class boundaries.
        code_max_lines: Maximum lines per code chunk regardless of token count.
        respect_document_structure: Whether to preserve document structure boundaries.
        markdown_split_headers: Whether to split markdown at header boundaries.

    Example:
        Semantic chunking configuration:

        >>> chunk_config = ChunkingConfig(
        ...     max_chunk_size=1024,
        ...     use_semantic_chunking=True,
        ...     semantic_threshold=0.8
        ... )

        Code-optimized chunking:

        >>> chunk_config = ChunkingConfig(
        ...     code_chunk_by_function=True,
        ...     code_max_lines=50,
        ...     respect_document_structure=True
        ... )
    """

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
    """Semantic caching configuration for query result optimization.

    Configures semantic caching to improve response times by storing and reusing
    results for similar queries. Uses embedding similarity to determine cache hits
    with adaptive threshold optimization to maintain target hit rates.

    Environment variables can be prefixed with CACHE_ (e.g., CACHE_TTL_SECONDS=7200).

    Attributes:
        enabled: Whether semantic caching is active.
        ttl_seconds: Time-to-live for cached entries in seconds.
        similarity_threshold: Minimum similarity score for cache hits (0-1).
        max_cache_size: Maximum number of entries to store in cache.
        target_hit_rate: Target cache hit rate for optimization (0-1).
        adaptive_threshold: Whether to automatically adjust similarity threshold.

    Example:
        High-performance caching:

        >>> cache_config = CacheConfig(
        ...     ttl_seconds=7200,  # 2 hours
        ...     similarity_threshold=0.95,  # Very similar queries only
        ...     target_hit_rate=0.4  # 40% hit rate target
        ... )

        Aggressive caching:

        >>> cache_config = CacheConfig(
        ...     similarity_threshold=0.9,  # Lower threshold
        ...     adaptive_threshold=True,  # Auto-optimize
        ...     max_cache_size=2000  # Larger cache
        ... )
    """

    model_config = ConfigDict(env_prefix="CACHE_")

    enabled: bool = Field(default=True)
    ttl_seconds: int = Field(default=3600)
    similarity_threshold: float = Field(default=0.97)
    max_cache_size: int = Field(default=1000)

    # Cache hit rate optimization
    target_hit_rate: float = Field(default=0.31)
    adaptive_threshold: bool = Field(default=True)


class ContextConfig(BaseSettings):
    """Context composition and retrieval configuration for LLM optimization.

    Configures how context is retrieved, filtered, and composed for optimal LLM
    consumption. Manages token limits, relevance thresholds, and quality controls
    to provide the most useful context within model constraints.

    Environment variables can be prefixed with CONTEXT_ (e.g., CONTEXT_MAX_CONTEXT_TOKENS=16000).

    Attributes:
        max_context_tokens: Maximum tokens available for context in LLM requests.
        reserve_tokens_for_response: Tokens to reserve for LLM response generation.
        default_top_k: Default number of top results to retrieve from vector search.
        min_relevance_score: Minimum similarity score to include in context.
        use_hierarchical_retrieval: Whether to use hierarchical search strategy.
        progressive_loading: Whether to load context progressively by relevance.
        remove_redundancy: Whether to filter out redundant/duplicate content.
        redundancy_threshold: Similarity threshold for identifying redundant content.

    Example:
        Conservative context for accuracy:

        >>> context_config = ContextConfig(
        ...     max_context_tokens=8000,
        ...     min_relevance_score=0.8,  # High relevance only
        ...     remove_redundancy=True
        ... )

        Comprehensive context for coverage:

        >>> context_config = ContextConfig(
        ...     default_top_k=20,
        ...     min_relevance_score=0.6,  # Lower threshold
        ...     progressive_loading=True
        ... )
    """

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
    """Document processing configuration for multi-format support.

    Configures document processing pipeline including supported file formats,
    metadata extraction, language detection, and content parsing options.
    Handles various document types from code files to PDFs with appropriate
    processing strategies.

    Environment variables can be prefixed with DOC_ (e.g., DOC_MAX_FILE_SIZE_MB=50).

    Attributes:
        file_patterns: List of glob patterns for supported file types.
        extract_metadata: Whether to extract file metadata (dates, authors, etc.).
        detect_language: Whether to detect and tag content language.
        parse_code_structure: Whether to parse code structure (functions, classes).
        max_file_size_mb: Maximum file size in MB to process.
        skip_binary_files: Whether to skip binary files automatically.

    Example:
        Code-focused processing:

        >>> doc_config = DocumentConfig(
        ...     file_patterns=["*.py", "*.js", "*.md"],
        ...     parse_code_structure=True,
        ...     extract_metadata=True
        ... )

        Document-heavy processing:

        >>> doc_config = DocumentConfig(
        ...     file_patterns=["*.pdf", "*.docx", "*.md", "*.txt"],
        ...     max_file_size_mb=200,
        ...     detect_language=True
        ... )
    """

    model_config = ConfigDict(env_prefix="DOC_")

    # Supported file patterns
    file_patterns: List[str] = Field(
        default_factory=lambda: [
            "*.md",
            "*.txt",
            "*.json",
            "*.yaml",
            "*.yml",
            "*.py",
            "*.js",
            "*.ts",
            "*.jsx",
            "*.tsx",
            "*.java",
            "*.go",
            "*.rs",
            "*.cpp",
            "*.c",
            "*.pdf",
            "*.docx",
            "*.doc",
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
    """Main RAG configuration class aggregating all component configurations.

    Central configuration class that combines all sub-configurations into a
    single, coherent configuration object. Supports loading from environment
    variables, .env files, and configuration files (JSON/YAML).

    Environment variables can be prefixed with RAG_ (e.g., RAG_DEBUG=true).

    Attributes:
        redis: Redis connection and pooling configuration.
        embedding: Embedding model and provider configuration.
        index: Vector index and HNSW algorithm configuration.
        chunking: Document chunking and segmentation configuration.
        cache: Semantic caching optimization configuration.
        context: Context retrieval and composition configuration.
        document: Document processing and format support configuration.
        server_name: MCP server name for identification.
        server_version: Server version for compatibility tracking.
        debug: Enable debug mode with verbose logging.
        data_dir: Directory for persistent data storage.
        index_dir: Directory for index files and metadata.

    Example:
        Default configuration:

        >>> config = RAGConfig()
        >>> config.redis.host = "localhost"
        >>> config.embedding.provider = "sentence-transformers"

        Environment-based configuration:

        >>> # Set RAG_DEBUG=true, REDIS_HOST=redis.prod.com
        >>> config = RAGConfig()
        >>> print(f"Debug: {config.debug}, Redis: {config.redis.host}")

        File-based configuration:

        >>> config = RAGConfig.from_file(Path("production.yaml"))
        >>> await server.initialize(config)
    """

    model_config = ConfigDict(env_prefix="RAG_", env_file=".env", env_file_encoding="utf-8")

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
        """Ensure required directories exist, creating them if necessary.

        Automatically creates data and index directories with proper permissions
        during configuration validation. Prevents runtime errors from missing
        directories.

        Args:
            v: Path object for the directory.

        Returns:
            The validated Path object after ensuring directory exists.

        Raises:
            PermissionError: If unable to create directory due to permissions.
            OSError: If directory creation fails for other reasons.
        """
        v.mkdir(parents=True, exist_ok=True)
        return v

    @classmethod
    def from_file(cls, config_path: Path) -> "RAGConfig":
        """Load configuration from JSON or YAML file.

        Supports both JSON and YAML configuration files with automatic format
        detection based on file extension. Provides a convenient way to manage
        complex configurations externally.

        Args:
            config_path: Path to the configuration file (.json, .yaml, or .yml).

        Returns:
            RAGConfig instance populated with values from the file.

        Raises:
            ValueError: If file format is not supported (not .json, .yaml, .yml).
            FileNotFoundError: If configuration file doesn't exist.
            json.JSONDecodeError: If JSON file is malformed.
            yaml.YAMLError: If YAML file is malformed.

        Example:
            JSON configuration:

            >>> # config.json contains: {"debug": true, "redis": {"host": "prod-redis"}}
            >>> config = RAGConfig.from_file(Path("config.json"))
            >>> config.debug
            True

            YAML configuration:

            >>> # config.yaml contains:
            >>> # debug: true
            >>> # redis:
            >>> #   host: prod-redis
            >>> #   port: 6380
            >>> config = RAGConfig.from_file(Path("config.yaml"))
            >>> config.redis.port
            6380
        """
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

        return cls.model_validate(data)
