"""Document indexer with hierarchical organization and comprehensive metadata
management.

This module provides document indexing capabilities with support for hierarchical
organization (concepts → sections → chunks), folder scanning, file change detection,
and comprehensive metadata tracking. It integrates with document processors,
embedding generators, and Redis vector storage for scalable indexing operations.

The indexer supports:
- Hierarchical document organization (3 levels)
- Incremental indexing with change detection
- Git metadata extraction
- Gitignore support
- Comprehensive document metadata
- Progress tracking and error handling

Example:
    Basic document indexing:

    >>> from eol.rag_context.indexer import DocumentIndexer
    >>> from eol.rag_context.config import RAGConfig
    >>>
    >>> config = RAGConfig()
    >>> indexer = DocumentIndexer(config, processor, embeddings, redis_store)
    >>> result = await indexer.index_folder("/path/to/docs")
    >>> print(f"Indexed {result.indexed_files} files with {result.total_chunks} chunks")

"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import gitignore_parser

from .config import RAGConfig
from .document_processor import DocumentProcessor, ProcessedDocument
from .embeddings import EmbeddingManager
from .redis_client import RedisVectorStore, VectorDocument

logger = logging.getLogger(__name__)


@dataclass
class IndexedSource:
    """Represents metadata for a completely indexed data source.

    Tracks comprehensive information about an indexed source including
    file counts, chunk statistics, and associated metadata.

    Attributes:
        source_id: Unique identifier for this indexed source.
        path: Filesystem path to the indexed directory or file.
        indexed_at: Unix timestamp when indexing was completed.
        file_count: Total number of files discovered in the source.
        total_chunks: Total number of chunks created from all files.
        metadata: Additional source-specific metadata (git info, patterns, etc.).
        indexed_files: Number of files successfully processed.

    Example:
        >>> source = IndexedSource(
        ...     source_id="project_docs_abc123",
        ...     path=Path("/project/docs"),
        ...     indexed_at=time.time(),
        ...     file_count=25,
        ...     total_chunks=1247
        ... )
        >>> print(f"Source {source.source_id} has {source.total_chunks} chunks")

    """

    source_id: str
    path: Path
    indexed_at: float
    file_count: int
    total_chunks: int
    metadata: dict[str, Any] = field(default_factory=dict)
    indexed_files: int = 0

    def __post_init__(self):
        """Ensure data consistency after initialization.

        Sets indexed_files to match file_count if not explicitly provided, maintaining
        backward compatibility and data consistency.

        """
        if self.indexed_files == 0 and self.file_count > 0:
            self.indexed_files = self.file_count


@dataclass
class IndexResult:
    """Result object from single file or batch indexing operations.

    Contains statistics and status information from indexing operations,
    including success metrics and any errors encountered during processing.

    Attributes:
        source_id: Unique identifier for the source being indexed.
        chunks: Number of chunks created from this operation.
        files: Number of files processed in this operation.
        errors: List of error messages encountered during indexing.
        metadata: Additional operation-specific metadata.
        file_count: Alias for files (compatibility).
        total_chunks: Alias for chunks (compatibility).
        indexed_files: Number of files successfully indexed.

    Example:
        >>> result = IndexResult(
        ...     source_id="docs_abc123",
        ...     chunks=45,
        ...     files=3,
        ...     errors=[]
        ... )
        >>> if not result.errors:
        ...     print(f"Successfully indexed {result.files} files")

    """

    source_id: str
    chunks: int = 0
    files: int = 0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Additional fields for compatibility
    file_count: int = 0
    total_chunks: int = 0
    indexed_files: int = 0

    def __post_init__(self):
        """Ensure consistency between field names for backward compatibility.

        Synchronizes duplicate fields (file_count/files, total_chunks/chunks,
        indexed_files/files) to maintain API compatibility with different naming
        conventions used across the codebase.

        """
        if self.file_count == 0 and self.files > 0:
            self.file_count = self.files
        if self.total_chunks == 0 and self.chunks > 0:
            self.total_chunks = self.chunks
        if self.indexed_files == 0 and self.files > 0:
            self.indexed_files = self.files


@dataclass
class DocumentMetadata:
    """Comprehensive metadata for precise document and chunk tracking.

    Stores detailed metadata for documents and chunks including source information,
    file properties, indexing details, hierarchical relationships, content metadata,
    location information, and version control data.

    Attributes:
        source_path: Absolute filesystem path to the source file.
        source_id: Unique identifier for the containing source.
        relative_path: Path relative to the indexing root directory.
        file_type: Document type (markdown, code, pdf, etc.).
        file_size: File size in bytes.
        file_hash: SHA-256 hash of file content for change detection.
        modified_time: Unix timestamp of last file modification.
        indexed_at: Unix timestamp when document was indexed.
        chunk_index: Zero-based position of chunk within document.
        total_chunks: Total number of chunks in the source document.
        hierarchy_level: Hierarchical level (1=concept, 2=section, 3=chunk).
        parent_chunk_id: ID of parent chunk in hierarchy (optional).
        language: Programming language for code files (optional).
        headers: Markdown header hierarchy (optional).
        section_title: Title of current section (optional).
        line_start: Starting line number in source file (optional).
        line_end: Ending line number in source file (optional).
        char_start: Starting character position (optional).
        char_end: Ending character position (optional).
        git_commit: Git commit hash when indexed (optional).
        git_branch: Git branch name when indexed (optional).
        git_remote: Git remote URL when indexed (optional).

    Example:
        >>> metadata = DocumentMetadata(
        ...     source_path="/project/src/main.py",
        ...     source_id="project_src_abc123",
        ...     relative_path="src/main.py",
        ...     file_type="code",
        ...     file_size=1024,
        ...     file_hash="abc123...",
        ...     modified_time=time.time(),
        ...     indexed_at=time.time(),
        ...     chunk_index=0,
        ...     total_chunks=5,
        ...     hierarchy_level=3
        ... )

    """

    # Source identification
    source_path: str  # Absolute path to source file
    source_id: str  # Unique source identifier
    relative_path: str  # Relative path from scan root

    # File metadata
    file_type: str  # markdown, code, pdf, etc.
    file_size: int  # Size in bytes
    file_hash: str  # Content hash for change detection
    modified_time: float  # Last modification timestamp

    # Indexing metadata
    indexed_at: float  # When indexed
    chunk_index: int  # Position in document
    total_chunks: int  # Total chunks from this document

    # Hierarchy metadata
    hierarchy_level: int  # 1=concept, 2=section, 3=chunk
    parent_chunk_id: str | None = None

    # Content metadata
    language: str | None = None  # For code files
    headers: list[str] | None = None  # For markdown
    section_title: str | None = None  # Current section

    # Location metadata for precise retrieval
    line_start: int | None = None  # Start line in source
    line_end: int | None = None  # End line in source
    char_start: int | None = None  # Start character position
    char_end: int | None = None  # End character position

    # Git metadata (if in git repo)
    git_commit: str | None = None
    git_branch: str | None = None
    git_remote: str | None = None


class FolderScanner:
    """Intelligent folder scanner with gitignore support and filtering.

    Provides comprehensive folder scanning capabilities with support for recursive
    traversal, gitignore parsing, file pattern matching, size filtering, and
    intelligent ignore patterns for development environments.

    The scanner automatically excludes common development artifacts like
    .git directories, node_modules, __pycache__, build outputs, and other
    non-essential files while respecting .gitignore files when present.

    Attributes:
        config: RAG configuration containing document processing settings.
        ignore_patterns: Set of glob patterns to ignore during scanning.
        scanned_sources: Cache of previously scanned source metadata.

    Example:
        >>> scanner = FolderScanner(config)
        >>> files = await scanner.scan_folder(
        ...     "/project",
        ...     recursive=True,
        ...     respect_gitignore=True
        ... )
        >>> print(f"Found {len(files)} files to index")

    """

    def __init__(self, config: RAGConfig):
        """Initialize folder scanner with configuration and ignore patterns.

        Args:
            config: RAG configuration containing document processing settings.

        """
        self.config = config
        self.ignore_patterns = self._default_ignore_patterns()
        self.scanned_sources: dict[str, IndexedSource] = {}

    def _default_ignore_patterns(self) -> set[str]:
        """Generate default file and directory patterns to ignore during scanning.

        Returns comprehensive ignore patterns for common development artifacts,
        version control directories, build outputs, caches, and temporary files.

        Returns:
            Set of glob patterns that should be ignored during folder scanning.

        Note:
            These patterns are applied in addition to .gitignore rules when
            gitignore support is enabled. Patterns use Python's pathlib.Path.match()
            syntax with ** for recursive matching.

        """
        return {
            "**/.git/**",
            "**/.svn/**",
            "**/.hg/**",
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/.pytest_cache/**",
            "**/.venv/**",
            "**/venv/**",
            "**/.env/**",
            "**/env/**",
            "**/dist/**",
            "**/build/**",
            "**/.DS_Store",
            "**/Thumbs.db",
            "**/*.log",
            "**/*.tmp",
            "**/*.temp",
            "**/.idea/**",
            "**/.vscode/**",
            "**/coverage/**",
            "**/.coverage",
        }

    def _should_ignore(self, path: Path, gitignore_matcher=None) -> bool:
        """Determine whether a file or directory should be ignored during scanning.

        Checks path against gitignore rules (if provided), default ignore patterns,
        and file size limits to determine if it should be excluded from indexing.

        Args:
            path: Path to check for exclusion.
            gitignore_matcher: Optional gitignore matcher from gitignore-parser.
                If provided, gitignore rules take precedence.

        Returns:
            True if the path should be ignored, False if it should be processed.

        Example:
            >>> scanner = FolderScanner(config)
            >>> should_skip = scanner._should_ignore(Path("node_modules/package.json"))
            >>> print(f"Skip file: {should_skip}")
            Skip file: True

        """
        path_str = str(path)

        # Check gitignore if available
        if gitignore_matcher and gitignore_matcher(path_str):
            return True

        # Check default patterns
        for pattern in self.ignore_patterns:
            if path.match(pattern):
                return True

        # Check file size
        if path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.config.document.max_file_size_mb:
                logger.debug(f"Ignoring large file ({size_mb:.2f}MB): {path}")
                return True

        return False

    def _get_git_metadata(self, path: Path) -> dict[str, Any]:
        """Extract Git repository metadata for version control tracking.

        Attempts to extract Git metadata including repository root, current commit
        hash, branch name, and remote URL. Gracefully handles non-git directories
        and Git command failures.

        Args:
            path: Path to check for Git metadata (file or directory).

        Returns:
            Dictionary containing Git metadata:
            - git_root: Absolute path to Git repository root
            - git_commit: Current commit SHA hash
            - git_branch: Current branch name
            - git_remote: Origin remote URL

            Returns empty dict if not in a Git repository or on Git command failure.

        Example:
            >>> scanner = FolderScanner(config)
            >>> git_info = scanner._get_git_metadata(Path("/project/src"))
            >>> if git_info:
            ...     print(f"Branch: {git_info['git_branch']}")

        """
        try:
            import subprocess

            # Find git root
            git_root = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=path.parent if path.is_file() else path,
                capture_output=True,
                text=True,
                check=False,
            )

            if git_root.returncode != 0:
                return {}

            git_root_path = git_root.stdout.strip()

            # Get current commit
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=git_root_path,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get current branch
            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=git_root_path,
                capture_output=True,
                text=True,
                check=False,
            )

            # Get remote URL
            remote = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=git_root_path,
                capture_output=True,
                text=True,
                check=False,
            )

            return {
                "git_root": git_root_path,
                "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
                "git_branch": branch.stdout.strip() if branch.returncode == 0 else None,
                "git_remote": remote.stdout.strip() if remote.returncode == 0 else None,
            }
        except Exception as e:
            logger.debug(f"Could not get git metadata: {e}")
            return {}

    async def scan_folder(
        self,
        folder_path: Path | str,
        recursive: bool = True,
        respect_gitignore: bool = True,
        file_patterns: list[str] | None = None,
    ) -> list[Path]:
        """Scan folder structure and return list of files suitable for indexing.

        Performs comprehensive folder scanning with intelligent filtering based on
        file patterns, gitignore rules, size limits, and default ignore patterns.
        Returns deduplicated, sorted list of files ready for processing.

        Args:
            folder_path: Root directory to scan (Path object or string path).
            recursive: Whether to scan subdirectories recursively.
            respect_gitignore: Whether to parse and respect .gitignore files.
            file_patterns: List of glob patterns to match. If None, uses patterns
                from configuration (e.g., ["*.py", "*.md", "*.js"]).

        Returns:
            Sorted list of Path objects for files that should be indexed.

        Raises:
            ValueError: If folder_path doesn't exist or is not a directory.

        Example:
            Scan Python and Markdown files:

            >>> scanner = FolderScanner(config)
            >>> files = await scanner.scan_folder(
            ...     "/project",
            ...     recursive=True,
            ...     file_patterns=["*.py", "*.md"]
            ... )
            >>> print(f"Found {len(files)} files")

            Respect gitignore in repository:

            >>> files = await scanner.scan_folder(
            ...     "/repo",
            ...     recursive=True,
            ...     respect_gitignore=True
            ... )

        """
        # Handle both Path and string inputs
        if isinstance(folder_path, str):
            folder_path = Path(folder_path)

        if not folder_path.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")

        if not folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")

        folder_path = folder_path.resolve()  # Get absolute path
        file_patterns = file_patterns or self.config.document.file_patterns

        # Load gitignore if requested
        gitignore_matcher = None
        if respect_gitignore:
            gitignore_path = folder_path / ".gitignore"
            if gitignore_path.exists():
                gitignore_matcher = gitignore_parser.parse_gitignore(gitignore_path)

        files_to_index = []

        # Scan for matching files
        for pattern in file_patterns:
            if recursive:
                paths = folder_path.rglob(pattern)
            else:
                paths = folder_path.glob(pattern)

            for path in paths:
                if path.is_file() and not self._should_ignore(path, gitignore_matcher):
                    files_to_index.append(path)

        # Remove duplicates and sort
        files_to_index = sorted(set(files_to_index))

        logger.info(f"Found {len(files_to_index)} files to index in {folder_path}")
        return files_to_index

    def generate_source_id(self, path: Path) -> str:
        """Generate deterministic unique identifier for a source path.

        Creates a reproducible source ID based on the absolute path using MD5 hashing.
        The same path will always generate the same source ID, enabling reliable
        source tracking and incremental updates.

        Args:
            path: Path object for which to generate source ID.

        Returns:
            16-character hexadecimal string uniquely identifying the path.

        Example:
            >>> scanner = FolderScanner(config)
            >>> source_id = scanner.generate_source_id(Path("/project/docs"))
            >>> print(f"Source ID: {source_id}")
            Source ID: a1b2c3d4e5f6g7h8

        """
        abs_path = str(path.resolve())
        return hashlib.md5(abs_path.encode(), usedforsecurity=False).hexdigest()[:16]


class DocumentIndexer:
    """Comprehensive document indexer with hierarchical organization and metadata
    tracking.

    Provides complete document indexing pipeline including document processing,
    hierarchical organization (concepts → sections → chunks), embedding generation,
    vector storage, and comprehensive metadata management. Supports incremental
    indexing with change detection and folder-level operations.

    The indexer creates a three-level hierarchy:
    1. Concepts (level 1): High-level document summaries and main ideas
    2. Sections (level 2): Logical document sections (headers, functions, etc.)
    3. Chunks (level 3): Individual text segments for detailed search

    Features:
    - Hierarchical document organization with parent-child relationships
    - Comprehensive metadata tracking including git information
    - Incremental indexing with file change detection
    - Progress callbacks for long-running operations
    - Statistical tracking and error handling
    - Source management and cleanup operations

    Attributes:
        config: RAG configuration with indexing parameters.
        processor: Document processor for content extraction.
        embeddings: Embedding manager for vector generation.
        redis: Redis vector store for data persistence.
        scanner: Folder scanner for file discovery.
        stats: Indexing statistics and metrics.

    Example:
        Basic indexing workflow:

        >>> indexer = DocumentIndexer(config, processor, embeddings, redis_store)
        >>>
        >>> # Index entire folder
        >>> result = await indexer.index_folder("/project/docs")
        >>> print(f"Indexed {result.indexed_files} files")
        >>>
        >>> # Index single file
        >>> file_result = await indexer.index_file("/project/readme.md")
        >>> print(f"Created {file_result.chunks} chunks")
        >>>
        >>> # Get statistics
        >>> stats = indexer.get_stats()
        >>> print(f"Total documents: {stats['total_documents']}")

    """

    def __init__(
        self,
        config: RAGConfig,
        document_processor: DocumentProcessor,
        embedding_manager: EmbeddingManager,
        redis_store: RedisVectorStore,
    ):
        """Initialize document indexer with required components.

        Args:
            config: RAG configuration with indexing parameters.
            document_processor: Document processor for content extraction.
            embedding_manager: Embedding manager for vector generation.
            redis_store: Redis vector store for data persistence.

        """
        self.config = config
        self.processor = document_processor
        self.embeddings = embedding_manager
        self.redis = redis_store
        self.scanner = FolderScanner(config)

        # Indexing statistics
        self.stats = {
            "documents_indexed": 0,
            "chunks_created": 0,
            "concepts_extracted": 0,
            "sections_created": 0,
            "errors": 0,
            "indexing_time": 0.0,
        }

    async def index_folder(
        self,
        folder_path: Path | str,
        source_id: str | None = None,
        recursive: bool = True,
        force_reindex: bool = False,
        progress_callback: callable | None = None,
    ) -> IndexedSource:
        """Index all supported documents in a folder with hierarchical organization.

        Performs comprehensive folder indexing including file discovery, document
        processing, hierarchical organization, embedding generation, and metadata
        tracking. Supports incremental indexing to skip unchanged files.

        The indexing process:
        1. Scans folder for supported files using configured patterns
        2. Extracts Git metadata for version control tracking
        3. Processes each file through document processor
        4. Creates hierarchical structure (concepts → sections → chunks)
        5. Generates embeddings for all content levels
        6. Stores documents and metadata in Redis
        7. Updates source metadata and statistics

        Args:
            folder_path: Directory to index (Path object or string path).
            source_id: Unique source identifier. If None, generated from path hash.
            recursive: Whether to scan subdirectories recursively.
            force_reindex: If True, reindex all files regardless of change status.
            progress_callback: Optional callback function for progress updates.
                Called with (current_file_index, total_files, current_file_path).

        Returns:
            IndexedSource object containing indexing results and metadata:
            - source_id: Unique identifier for this indexed source
            - path: Absolute path to the indexed directory
            - indexed_at: Timestamp of indexing completion
            - file_count: Total files discovered in directory
            - total_chunks: Total chunks created from all files
            - indexed_files: Number of files actually processed
            - metadata: Additional information (git info, patterns, etc.)

        Example:
            Index entire project:

            >>> def progress(current, total, file_path):
            ...     print(f"Processing {current}/{total}: {file_path.name}")
            >>>
            >>> result = await indexer.index_folder(
            ...     "/project",
            ...     recursive=True,
            ...     progress_callback=progress
            ... )
            >>> print(f"Indexed {result.indexed_files} files")
            >>> print(f"Created {result.total_chunks} chunks")

            Force reindex with custom source ID:

            >>> result = await indexer.index_folder(
            ...     "/docs",
            ...     source_id="docs_v2",
            ...     force_reindex=True
            ... )

        """
        start_time = time.time()
        # Handle both Path and string inputs
        if isinstance(folder_path, str):
            folder_path = Path(folder_path)
        folder_path = folder_path.resolve()

        # Generate source ID if not provided
        if not source_id:
            source_id = self.scanner.generate_source_id(folder_path)

        # Scan folder
        files = await self.scanner.scan_folder(folder_path, recursive=recursive)

        # Get git metadata for the folder
        git_metadata = self.scanner._get_git_metadata(folder_path)

        # Index each file
        total_chunks = 0
        indexed_files = 0
        all_errors = []
        skipped_files = 0

        # Get existing source info if available
        existing = await self._get_indexed_source(source_id)
        if existing and not force_reindex:
            # Start with existing chunk count for unchanged files
            total_chunks = existing.total_chunks

        for i, file_path in enumerate(files):
            try:
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, len(files), file_path)

                # Check if file needs reindexing
                if not force_reindex and await self._is_file_current(file_path):
                    logger.debug(f"Skipping unchanged file: {file_path}")
                    skipped_files += 1
                    continue

                # Process and index file
                result = await self.index_file(
                    file_path,
                    source_id=source_id,
                    root_path=folder_path,
                    git_metadata=git_metadata,
                )

                total_chunks += result.chunks
                indexed_files += 1

                # Collect any errors
                if result.errors:
                    all_errors.extend(result.errors)

            except Exception as e:
                logger.error(f"Error indexing {file_path}: {e}")
                self.stats["errors"] += 1
                all_errors.append(f"Error indexing {file_path}: {str(e)}")

        # Store source metadata
        indexed_source = IndexedSource(
            source_id=source_id,
            path=folder_path,
            indexed_at=time.time(),
            file_count=len(files),  # Total files in folder
            total_chunks=total_chunks,
            indexed_files=indexed_files + skipped_files,  # All files processed
            metadata={
                "recursive": recursive,
                "git": git_metadata,
                "patterns": self.config.document.file_patterns,
            },
        )

        await self._store_indexed_source(indexed_source)

        # Update statistics
        self.stats["indexing_time"] += time.time() - start_time

        logger.info(f"Indexed {indexed_files} files, {total_chunks} chunks from {folder_path}")
        return indexed_source

    async def index_file(
        self,
        file_path: Path | str,
        source_id: str | None = None,
        root_path: Path | None = None,
        git_metadata: dict[str, Any] | None = None,
    ) -> IndexResult:
        """Index a single file with comprehensive hierarchical organization and
        metadata.

        Processes a single file through the complete indexing pipeline including
        document processing, content extraction, hierarchical structuring,
        embedding generation, and storage with full metadata tracking.

        The file indexing process:
        1. Processes file content using document processor
        2. Extracts file metadata (size, hash, modification time)
        3. Creates document hierarchy (concepts → sections → chunks)
        4. Generates embeddings for all content levels
        5. Stores documents with comprehensive metadata
        6. Updates file change tracking information

        Args:
            file_path: Path to file to index (Path object or string path).
            source_id: Unique source identifier. If None, generated from parent path.
            root_path: Root directory for relative path calculation. If None,
                uses file's parent directory.
            git_metadata: Pre-extracted Git metadata. If None, Git information
                is not included in document metadata.

        Returns:
            IndexResult object containing operation results:
            - source_id: Unique identifier for the source
            - chunks: Number of chunks created from this file
            - files: Number of files processed (always 1)
            - errors: List of any errors encountered
            - metadata: Additional file-specific information

        Example:
            Index single file:

            >>> result = await indexer.index_file("/project/readme.md")
            >>> if not result.errors:
            ...     print(f"Created {result.chunks} chunks")
            ... else:
            ...     print(f"Errors: {result.errors}")

            Index with Git metadata:

            >>> git_info = scanner._get_git_metadata(Path("/project"))
            >>> result = await indexer.index_file(
            ...     "/project/src/main.py",
            ...     git_metadata=git_info
            ... )

        """
        # Handle both Path and string inputs
        if isinstance(file_path, str):
            file_path = Path(file_path)
        file_path = file_path.resolve()

        # Generate source ID if not provided
        if not source_id:
            source_id = self.scanner.generate_source_id(file_path.parent)

        # Initialize result
        result = IndexResult(source_id=source_id, files=1, file_count=1, indexed_files=1)

        try:
            # Calculate relative path
            if root_path:
                relative_path = file_path.relative_to(root_path)
            else:
                relative_path = file_path.name

            # Process document
            doc = await self.processor.process_file(file_path)
            if not doc:
                return result

            # Calculate file hash
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            # Get file metadata
            file_stat = file_path.stat()

            # Create base metadata
            base_metadata = DocumentMetadata(
                source_path=str(file_path),
                source_id=source_id,
                relative_path=str(relative_path),
                file_type=doc.doc_type,
                file_size=file_stat.st_size,
                file_hash=file_hash,
                modified_time=file_stat.st_mtime,
                indexed_at=time.time(),
                chunk_index=0,
                total_chunks=len(doc.chunks),
                hierarchy_level=3,  # Start with chunks
                language=doc.language,
                git_commit=git_metadata.get("git_commit") if git_metadata else None,
                git_branch=git_metadata.get("git_branch") if git_metadata else None,
                git_remote=git_metadata.get("git_remote") if git_metadata else None,
            )

            # Extract document-level concepts (hierarchy level 1)
            concepts = await self._extract_concepts(doc, base_metadata)

            # Extract sections (hierarchy level 2)
            sections = await self._extract_sections(doc, base_metadata, concepts)

            # Index chunks (hierarchy level 3)
            chunks_indexed = await self._index_chunks(doc, base_metadata, sections)

            # Update statistics
            self.stats["documents_indexed"] += 1
            self.stats["chunks_created"] += chunks_indexed
            self.stats["concepts_extracted"] += len(concepts)
            self.stats["sections_created"] += len(sections)

            # Store file metadata for change detection
            await self._store_file_metadata(file_path, file_hash, file_stat.st_mtime)

            # Update result
            result.chunks = chunks_indexed
            result.total_chunks = chunks_indexed
            result.metadata = {
                "file_path": str(file_path),
                "concepts": len(concepts),
                "sections": len(sections),
            }

            return result

        except Exception as e:
            # Add error to result
            result.errors.append(f"Error indexing {file_path}: {str(e)}")
            logger.error(f"Error indexing {file_path}: {e}")
            return result

    async def _extract_concepts(
        self, doc: ProcessedDocument, base_metadata: DocumentMetadata
    ) -> list[VectorDocument]:
        """Extract high-level concepts from document for hierarchy level 1.

        Creates concept-level documents that represent the main ideas and themes
        of the source document. Currently generates a single concept based on
        document summary, but can be extended for multiple concept extraction.

        Args:
            doc: Processed document containing content and metadata.
            base_metadata: Base metadata template for the document.

        Returns:
            List of VectorDocument objects representing document concepts.
            Each concept includes summary content, embeddings, and metadata.

        """
        concepts = []

        # Generate document summary as main concept
        summary = self._generate_summary(doc.content[:2000])  # First 2000 chars

        # Create concept document
        concept_id = f"{base_metadata.source_id}_concept_main"
        # Filter out None values for Redis
        concept_metadata = {k: v for k, v in asdict(base_metadata).items() if v is not None}
        concept_metadata.update(
            {
                "hierarchy_level": 1,
                "concept_type": "document_summary",
                "file_path": str(doc.file_path),
            }
        )

        # Generate embedding
        embedding = await self.embeddings.get_embedding(summary)

        concept = VectorDocument(
            id=concept_id,
            content=summary,
            embedding=embedding,
            metadata=concept_metadata,
            hierarchy_level=1,
            children_ids=[],  # Will be populated with section IDs
        )

        # Store concept
        await self.redis.store_document(concept)
        concepts.append(concept)

        return concepts

    async def _extract_sections(
        self,
        doc: ProcessedDocument,
        base_metadata: DocumentMetadata,
        concepts: list[VectorDocument],
    ) -> list[VectorDocument]:
        """Extract logical sections from document for hierarchy level 2.

        Creates section-level documents by grouping related chunks based on
        document structure. Uses different strategies based on document type:
        - Markdown: Groups by header structure
        - Code: Groups by functions/classes
        - Other: Groups by chunk proximity

        Args:
            doc: Processed document containing chunks and metadata.
            base_metadata: Base metadata template for the document.
            concepts: Parent concept documents for relationship linking.

        Returns:
            List of VectorDocument objects representing document sections.
            Each section contains grouped content, embeddings, and parent links.

        """
        sections = []

        # Group chunks into logical sections
        if doc.doc_type == "markdown" and doc.metadata.get("headers"):
            # Use header structure for sections
            current_section = []
            current_header = None

            for chunk in doc.chunks:
                if chunk.get("header") != current_header:
                    # Save previous section
                    if current_section:
                        section = await self._create_section(
                            current_section,
                            base_metadata,
                            concepts[0].id if concepts else None,
                            current_header,
                        )
                        sections.append(section)

                    # Start new section
                    current_header = chunk.get("header")
                    current_section = [chunk]
                else:
                    current_section.append(chunk)

            # Save final section
            if current_section:
                section = await self._create_section(
                    current_section,
                    base_metadata,
                    concepts[0].id if concepts else None,
                    current_header,
                )
                sections.append(section)

        elif doc.doc_type == "code":
            # Group by functions/classes for code
            for i, chunk in enumerate(doc.chunks):
                if chunk.get("type") in [
                    "function_definition",
                    "class_definition",
                    "method_declaration",
                ]:
                    section = await self._create_section(
                        [chunk],
                        base_metadata,
                        concepts[0].id if concepts else None,
                        chunk.get("type"),
                    )
                    sections.append(section)

        else:
            # Create sections based on chunk size
            section_size = 5  # Group 5 chunks per section
            for i in range(0, len(doc.chunks), section_size):
                section_chunks = doc.chunks[i : i + section_size]
                section = await self._create_section(
                    section_chunks,
                    base_metadata,
                    concepts[0].id if concepts else None,
                    f"Section {i // section_size + 1}",
                )
                sections.append(section)

        # Update concept with section IDs
        if concepts and sections:
            concepts[0].children_ids = [s.id for s in sections]
            await self.redis.store_document(concepts[0])

        return sections

    async def _create_section(
        self,
        chunks: list[dict[str, Any]],
        base_metadata: DocumentMetadata,
        parent_id: str | None,
        section_title: str | None,
    ) -> VectorDocument:
        """Create section-level document from grouped chunks.

        Combines multiple related chunks into a section-level document with
        appropriate metadata, embeddings, and hierarchical relationships.

        Args:
            chunks: List of chunk dictionaries to combine into section.
            base_metadata: Base metadata template for the section.
            parent_id: ID of parent concept document.
            section_title: Title or identifier for this section.

        Returns:
            VectorDocument representing the section with combined content.

        """
        # Combine chunk content
        content = "\n\n".join([c["content"] for c in chunks])

        # Generate section ID
        content_hash = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:8]
        section_id = f"{base_metadata.source_id}_section_{content_hash}"

        # Create section metadata - filter out None values for Redis
        section_metadata = {k: v for k, v in asdict(base_metadata).items() if v is not None}
        section_metadata.update(
            {
                "hierarchy_level": 2,
                "section_title": section_title,
                "chunk_count": len(chunks),
            }
        )

        # Generate embedding
        embedding = await self.embeddings.get_embedding(content[:1000])  # Use first 1000 chars

        section = VectorDocument(
            id=section_id,
            content=content[:1000],  # Store truncated content
            embedding=embedding,
            metadata=section_metadata,
            hierarchy_level=2,
            parent_id=parent_id,
            children_ids=[],  # Will be populated with chunk IDs
        )

        # Store section
        await self.redis.store_document(section)

        return section

    async def _index_chunks(
        self,
        doc: ProcessedDocument,
        base_metadata: DocumentMetadata,
        sections: list[VectorDocument],
    ) -> int:
        """Index document chunks at hierarchy level 3 with comprehensive metadata.

        Processes individual document chunks, generates embeddings, creates
        comprehensive metadata, establishes parent-child relationships with
        sections, and stores everything in Redis vector storage.

        Args:
            doc: Processed document containing chunks to index.
            base_metadata: Base metadata template for all chunks.
            sections: Parent section documents for relationship linking.

        Returns:
            Number of chunks successfully indexed.

        """
        chunks_indexed = 0
        section_idx = 0
        chunks_per_section = max(1, len(doc.chunks) // max(1, len(sections)))

        for i, chunk in enumerate(doc.chunks):
            # Determine parent section
            if sections:
                section_idx = min(i // chunks_per_section, len(sections) - 1)
                parent_section = sections[section_idx]
            else:
                parent_section = None

            # Create chunk metadata - filter out None values for Redis
            chunk_metadata = {k: v for k, v in asdict(base_metadata).items() if v is not None}
            chunk_updates = {
                "chunk_index": i,
                "hierarchy_level": 3,
                "section_title": chunk.get("header") or chunk.get("type"),
                "line_start": chunk.get("start_line"),
                "line_end": chunk.get("end_line"),
                "char_start": chunk.get("char_start"),
                "char_end": chunk.get("char_end"),
            }
            # Only add non-None values
            chunk_metadata.update({k: v for k, v in chunk_updates.items() if v is not None})

            # Add chunk-specific metadata
            if doc.doc_type == "code":
                chunk_metadata["function_name"] = chunk.get("function_name")
                chunk_metadata["class_name"] = chunk.get("class_name")
            elif doc.doc_type == "markdown":
                chunk_metadata["headers"] = doc.metadata.get("headers", [])

            # Generate chunk ID
            chunk_id = f"{base_metadata.source_id}_chunk_{i}"

            # Generate embedding
            embedding = await self.embeddings.get_embedding(chunk["content"])

            # Create vector document
            vector_doc = VectorDocument(
                id=chunk_id,
                content=chunk["content"],
                embedding=embedding,
                metadata=chunk_metadata,
                hierarchy_level=3,
                parent_id=parent_section.id if parent_section else None,
            )

            # Store in Redis
            await self.redis.store_document(vector_doc)

            # Update parent section's children
            if parent_section:
                parent_section.children_ids.append(chunk_id)

            chunks_indexed += 1

        # Update sections with their chunk IDs
        for section in sections:
            if section.children_ids:
                await self.redis.store_document(section)

        return chunks_indexed

    def _generate_summary(self, content: str) -> str:
        """Generate extractive summary of document content for concept creation.

        Creates a concise summary by extracting the most meaningful lines from
        the beginning of the document content. Uses simple heuristics to select
        substantive content while avoiding very short or empty lines.

        Args:
            content: Raw document content to summarize.

        Returns:
            Summary string of approximately 500 characters containing key
            content from the document, or truncated content if extraction fails.

        """
        # Simple extractive summary - take first paragraph and key sentences
        lines = content.split("\n")
        summary_lines = []

        for line in lines[:10]:  # First 10 lines
            if len(line.strip()) > 20:  # Skip very short lines
                summary_lines.append(line.strip())
                if len(" ".join(summary_lines)) > 500:
                    break

        return " ".join(summary_lines) or content[:500]

    async def _is_file_current(self, file_path: Path) -> bool:
        """Check if file has been modified since last indexing for incremental updates.

        Compares current file modification time with stored metadata to determine
        if the file needs reindexing. Supports incremental indexing by skipping
        unchanged files to improve performance on large document collections.

        Args:
            file_path: Path to file to check for modifications.

        Returns:
            True if file is current (not modified since last indexing),
            False if file needs reindexing or has no stored metadata.

        """
        try:
            # Get stored metadata
            path_hash = hashlib.md5(str(file_path).encode(), usedforsecurity=False).hexdigest()
            file_key = f"file_meta:{path_hash}"
            stored = self.redis.redis.hgetall(file_key)

            if not stored:
                return False

            # Check modification time
            current_mtime = file_path.stat().st_mtime
            stored_mtime = float(stored.get(b"mtime", 0))

            return current_mtime <= stored_mtime
        except Exception as e:
            logger.debug(f"Error checking file currency: {e}")
            return False

    async def _store_file_metadata(self, file_path: Path, file_hash: str, mtime: float) -> None:
        """Store file metadata in Redis for change detection and incremental indexing.

        Persists file metadata including path, content hash, modification time,
        and indexing timestamp to enable efficient change detection for future
        indexing operations. Metadata expires after 30 days.

        Args:
            file_path: Path to file being tracked.
            file_hash: SHA-256 hash of file content.
            mtime: File modification timestamp.

        """
        file_hash = hashlib.md5(str(file_path).encode(), usedforsecurity=False).hexdigest()
        file_key = f"file_meta:{file_hash}"

        self.redis.redis.hset(
            file_key,
            mapping={
                "path": str(file_path),
                "hash": file_hash,
                "mtime": mtime,
                "indexed_at": time.time(),
            },
        )

        # Set TTL to 30 days
        self.redis.redis.expire(file_key, 30 * 24 * 3600)

    async def _get_indexed_source(self, source_id: str) -> IndexedSource | None:
        """Retrieve indexed source metadata from Redis storage.

        Fetches complete source metadata including statistics and configuration
        for the specified source ID. Used for incremental indexing and source
        management operations.

        Args:
            source_id: Unique identifier of the source to retrieve.

        Returns:
            IndexedSource object if found, None if source doesn't exist.

        """
        source_key = f"source:{source_id}"
        data = self.redis.redis.hgetall(source_key)

        if not data:
            return None

        return IndexedSource(
            source_id=source_id,
            path=Path(data[b"path"].decode()),
            indexed_at=float(data[b"indexed_at"]),
            file_count=int(data[b"file_count"]),
            total_chunks=int(data[b"total_chunks"]),
            indexed_files=int(data.get(b"indexed_files", data[b"file_count"])),
            metadata=(json.loads(data[b"metadata"].decode()) if data.get(b"metadata") else {}),
        )

    async def _store_indexed_source(self, source: IndexedSource) -> None:
        """Persist indexed source metadata to Redis storage.

        Stores complete source information including statistics, timestamps,
        and configuration metadata for source management and tracking.

        Args:
            source: IndexedSource object containing metadata to persist.

        """
        source_key = f"source:{source.source_id}"

        self.redis.redis.hset(
            source_key,
            mapping={
                "path": str(source.path),
                "indexed_at": source.indexed_at,
                "file_count": source.file_count,
                "total_chunks": source.total_chunks,
                "indexed_files": source.indexed_files,
                "metadata": json.dumps(source.metadata) if source.metadata else "{}",
            },
        )

    async def list_sources(self) -> list[IndexedSource]:
        """Retrieve list of all indexed sources with their metadata.

        Scans Redis for all source metadata and returns complete IndexedSource
        objects with statistics and metadata for each indexed source.

        Returns:
            List of IndexedSource objects containing:
            - source_id: Unique identifier
            - path: Original indexed path
            - indexed_at: Indexing timestamp
            - file_count: Number of files in source
            - total_chunks: Total chunks created
            - indexed_files: Files successfully processed
            - metadata: Additional source information

        Example:
            >>> sources = await indexer.list_sources()
            >>> for source in sources:
            ...     print(f"{source.source_id}: {source.file_count} files")
            ...     print(f"  Path: {source.path}")
            ...     print(f"  Chunks: {source.total_chunks}")

        """
        sources = []

        # Scan for source keys
        cursor = 0
        while True:
            cursor, keys = self.redis.redis.scan(cursor, match="source:*", count=100)

            for key in keys:
                source_id = key.decode().split(":", 1)[1]
                source = await self._get_indexed_source(source_id)
                if source:
                    sources.append(source)

            if cursor == 0:
                break

        return sources

    async def remove_source(self, source_id: str) -> bool:
        """Completely remove an indexed source and all associated data.

        Removes all documents, metadata, and tracking information for the specified
        source from Redis storage. This includes all hierarchy levels (concepts,
        sections, chunks) and file metadata used for change detection.

        Args:
            source_id: Unique identifier of the source to remove.

        Returns:
            True if any data was removed, False if source was not found.

        Example:
            >>> success = await indexer.remove_source("docs_abc123")
            >>> if success:
            ...     print("Source removed successfully")
            ... else:
            ...     print("Source not found")

        """
        # Delete all documents from this source
        deleted = 0

        for prefix in ["concept:", "section:", "chunk:"]:
            cursor = 0
            while True:
                cursor, keys = self.redis.redis.scan(
                    cursor, match=f"{prefix}{source_id}*", count=100
                )

                if keys:
                    self.redis.redis.delete(*keys)
                    deleted += len(keys)

                if cursor == 0:
                    break

        # Delete source metadata
        source_key = f"source:{source_id}"
        deleted += self.redis.redis.delete(source_key)

        # Delete file metadata
        cursor = 0
        while True:
            cursor, keys = self.redis.redis.scan(cursor, match="file_meta:*", count=100)

            for key in keys:
                data = self.redis.redis.hgetall(key)
                if data and source_id in data.get(b"path", b"").decode():
                    self.redis.redis.delete(key)
                    deleted += 1

            if cursor == 0:
                break

        logger.info(f"Removed source {source_id}, deleted {deleted} items")
        return deleted > 0

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive indexing statistics and metrics.

        Returns detailed statistics about indexing operations including document
        counts, chunk statistics, hierarchy metrics, error counts, timing
        information, and source counts from Redis.

        Returns:
            Dictionary containing indexing statistics:
            - documents_indexed: Total number of documents processed
            - chunks_created: Total number of chunks created
            - concepts_extracted: Number of concept-level documents
            - sections_created: Number of section-level documents
            - errors: Number of errors encountered during indexing
            - indexing_time: Total time spent on indexing operations
            - total_documents: Alias for documents_indexed (compatibility)
            - total_chunks: Alias for chunks_created (compatibility)
            - sources: Number of indexed sources in Redis

        Example:
            >>> stats = indexer.get_stats()
            >>> print(f"Indexed {stats['total_documents']} documents")
            >>> print(f"Created {stats['total_chunks']} chunks")
            >>> print(f"Errors: {stats['errors']}")
            >>> print(f"Time: {stats['indexing_time']:.2f}s")

        """
        stats = self.stats.copy()
        # Add required fields for compatibility
        stats["total_documents"] = stats.get("documents_indexed", 0)
        stats["total_chunks"] = stats.get("chunks_created", 0)

        # Count sources
        try:
            # Simple approximation - count unique source keys in Redis
            cursor = 0
            source_count = 0
            while True:
                cursor, keys = self.redis.redis.scan(cursor, match="source:*", count=100)
                source_count += len(keys)
                if cursor == 0:
                    break
            stats["sources"] = source_count
        except Exception:
            stats["sources"] = 0

        return stats

    async def index_file_dict(
        self, file_path: Path | str, source_id: str | None = None
    ) -> dict[str, Any]:
        """Index a file and return results as dictionary for API compatibility.

        Wrapper method that calls index_file() and converts the IndexResult
        to a dictionary format for backward compatibility with existing APIs
        and test suites that expect dictionary responses.

        Args:
            file_path: Path to file to index (Path object or string path).
            source_id: Optional unique source identifier.

        Returns:
            Dictionary containing indexing results:
            - status: "success" (always for this method)
            - source_id: Unique identifier for the source
            - chunks: Number of chunks created
            - total_chunks: Same as chunks (compatibility)
            - files: Number of files processed (always 1)
            - errors: List of any errors encountered
            - metadata: Additional file-specific information

        Example:
            >>> result = await indexer.index_file_dict("/project/readme.md")
            >>> print(f"Status: {result['status']}")
            >>> print(f"Chunks: {result['chunks']}")

        """
        result = await self.index_file(file_path, source_id)
        return {
            "status": "success",
            "source_id": result.source_id,
            "chunks": result.chunks,
            "total_chunks": result.total_chunks,
            "files": result.files,
            "errors": result.errors,
            "metadata": result.metadata,
        }
