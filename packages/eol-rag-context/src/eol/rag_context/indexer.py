"""
Document indexer with folder scanning and metadata management.
"""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
import numpy as np
import logging
from datetime import datetime
import gitignore_parser

from .config import RAGConfig
from .document_processor import DocumentProcessor, ProcessedDocument
from .embeddings import EmbeddingManager
from .redis_client import RedisVectorStore, VectorDocument

logger = logging.getLogger(__name__)


@dataclass
class IndexedSource:
    """Represents an indexed data source."""

    source_id: str
    path: Path
    indexed_at: float
    file_count: int
    total_chunks: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    indexed_files: int = 0

    def __post_init__(self):
        """Ensure indexed_files matches file_count if not set."""
        if self.indexed_files == 0 and self.file_count > 0:
            self.indexed_files = self.file_count


@dataclass
class IndexResult:
    """Result from indexing operations."""

    source_id: str
    chunks: int = 0
    files: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Additional fields for compatibility
    file_count: int = 0
    total_chunks: int = 0
    indexed_files: int = 0

    def __post_init__(self):
        """Ensure consistency between field names."""
        if self.file_count == 0 and self.files > 0:
            self.file_count = self.files
        if self.total_chunks == 0 and self.chunks > 0:
            self.total_chunks = self.chunks
        if self.indexed_files == 0 and self.files > 0:
            self.indexed_files = self.files


@dataclass
class DocumentMetadata:
    """Comprehensive metadata for document tracking."""

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
    parent_chunk_id: Optional[str] = None

    # Content metadata
    language: Optional[str] = None  # For code files
    headers: Optional[List[str]] = None  # For markdown
    section_title: Optional[str] = None  # Current section

    # Location metadata for precise retrieval
    line_start: Optional[int] = None  # Start line in source
    line_end: Optional[int] = None  # End line in source
    char_start: Optional[int] = None  # Start character position
    char_end: Optional[int] = None  # End character position

    # Git metadata (if in git repo)
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_remote: Optional[str] = None


class FolderScanner:
    """Scans folder structures and manages indexing."""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.ignore_patterns = self._default_ignore_patterns()
        self.scanned_sources: Dict[str, IndexedSource] = {}

    def _default_ignore_patterns(self) -> Set[str]:
        """Default patterns to ignore during scanning."""
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
        """Check if path should be ignored."""
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

    def _get_git_metadata(self, path: Path) -> Dict[str, Any]:
        """Extract git metadata if path is in a git repository."""
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
        file_patterns: Optional[List[str]] = None,
    ) -> List[Path]:
        """
        Scan folder for files to index.

        Args:
            folder_path: Root folder to scan
            recursive: Whether to scan recursively
            respect_gitignore: Whether to respect .gitignore files
            file_patterns: Specific patterns to match (uses config if None)

        Returns:
            List of file paths to index
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
        """Generate unique source ID for a path."""
        abs_path = str(path.resolve())
        return hashlib.md5(abs_path.encode()).hexdigest()[:16]


class DocumentIndexer:
    """Indexes documents with comprehensive metadata tracking."""

    def __init__(
        self,
        config: RAGConfig,
        document_processor: DocumentProcessor,
        embedding_manager: EmbeddingManager,
        redis_store: RedisVectorStore,
    ):
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
        source_id: Optional[str] = None,
        recursive: bool = True,
        force_reindex: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> IndexedSource:
        """
        Index all documents in a folder.

        Args:
            folder_path: Folder to index
            source_id: Optional source ID (generated if not provided)
            recursive: Whether to scan recursively
            force_reindex: Force reindexing even if unchanged
            progress_callback: Callback for progress updates

        Returns:
            IndexedSource summary
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
                    file_path, source_id=source_id, root_path=folder_path, git_metadata=git_metadata
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
        source_id: Optional[str] = None,
        root_path: Optional[Path] = None,
        git_metadata: Optional[Dict[str, Any]] = None,
    ) -> IndexResult:
        """
        Index a single file with full metadata tracking.

        Returns:
            IndexResult with source_id, chunks count, and any errors
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
    ) -> List[VectorDocument]:
        """Extract high-level concepts from document."""
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
        concepts: List[VectorDocument],
    ) -> List[VectorDocument]:
        """Extract document sections."""
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
        chunks: List[Dict[str, Any]],
        base_metadata: DocumentMetadata,
        parent_id: Optional[str],
        section_title: Optional[str],
    ) -> VectorDocument:
        """Create a section from chunks."""
        # Combine chunk content
        content = "\n\n".join([c["content"] for c in chunks])

        # Generate section ID
        section_id = (
            f"{base_metadata.source_id}_section_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        )

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
        sections: List[VectorDocument],
    ) -> int:
        """Index document chunks with full metadata."""
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
        """Generate a summary of content for concept extraction."""
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
        """Check if file has been modified since last indexing."""
        try:
            # Get stored metadata
            file_key = f"file_meta:{hashlib.md5(str(file_path).encode()).hexdigest()}"
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
        """Store file metadata for change detection."""
        file_key = f"file_meta:{hashlib.md5(str(file_path).encode()).hexdigest()}"

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

    async def _get_indexed_source(self, source_id: str) -> Optional[IndexedSource]:
        """Get indexed source metadata."""
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
            metadata=json.loads(data[b"metadata"].decode()) if data.get(b"metadata") else {},
        )

    async def _store_indexed_source(self, source: IndexedSource) -> None:
        """Store indexed source metadata."""
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

    async def list_sources(self) -> List[IndexedSource]:
        """List all indexed sources."""
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
        """Remove an indexed source and all its documents."""
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
        self.redis.redis.delete(source_key)

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

    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
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
        except:
            stats["sources"] = 0

        return stats

    async def index_file_dict(
        self, file_path: Path | str, source_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Index a file and return dict for compatibility.
        This is a wrapper for tests that expect dict format.
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
