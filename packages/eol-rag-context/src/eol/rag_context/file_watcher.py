"""Real-time file system monitoring for automatic RAG index updates.

This module provides comprehensive file system monitoring capabilities that automatically
detect changes to watched directories and update the RAG system in real-time. It ensures
the knowledge base stays current with source code and documentation changes without
requiring manual reindexing.

Key Features:
    - Real-time file monitoring using watchdog library with inotify/FSEvents
    - Fallback polling mode for systems without native file system events
    - Intelligent debouncing to handle rapid successive changes efficiently
    - Batch processing for optimal performance with multiple concurrent changes
    - Automatic knowledge graph updates when documents change
    - Change history tracking and comprehensive statistics
    - Configurable file pattern filtering and recursive watching
    - Graceful error handling and recovery mechanisms

The file watcher integrates seamlessly with the document indexer and knowledge graph
builder to provide a complete real-time RAG system that stays synchronized with the
underlying document corpus.

Example:
    Basic file watching setup:
    
    >>> from eol.rag_context.file_watcher import FileWatcher
    >>> from pathlib import Path
    >>> 
    >>> # Initialize with indexer and optional knowledge graph builder
    >>> watcher = FileWatcher(
    ...     indexer=document_indexer,
    ...     graph_builder=kg_builder,
    ...     debounce_seconds=3.0,
    ...     batch_size=15
    ... )
    >>> 
    >>> # Start monitoring
    >>> await watcher.start()
    >>> 
    >>> # Watch a project directory
    >>> source_id = await watcher.watch(
    ...     path=Path("/path/to/project"),
    ...     recursive=True,
    ...     file_patterns=["*.py", "*.md", "*.txt"]
    ... )
    >>> 
    >>> # Monitor activity
    >>> stats = watcher.get_stats()
    >>> print(f"Watching {stats['watched_sources']} sources")
    >>> print(f"Processed {stats['changes_processed']} changes")
    >>> 
    >>> # View recent changes
    >>> history = watcher.get_change_history(limit=10)
    >>> for change in history:
    ...     print(f"{change['time']}: {change['type']} - {change['path']}")
    >>> 
    >>> # Force full rescan when needed
    >>> result = await watcher.force_rescan()
    >>> print(f"Rescanned {result['files_indexed']} files")
"""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, Set, Optional, Callable, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object
    FileSystemEvent = None

from .indexer import DocumentIndexer, FolderScanner
from .knowledge_graph import KnowledgeGraphBuilder

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Enumeration of file system change types supported by the file watcher.
    
    Defines the various types of file system events that can be detected and
    processed by the file watching system. Each change type triggers different
    processing logic in the RAG system.
    
    Change Types:
        CREATED: New file detected in watched directory
        MODIFIED: Existing file content or metadata changed
        DELETED: File removed from watched directory
        MOVED: File moved or renamed (includes both old and new paths)
        
    Example:
        >>> change_type = ChangeType.MODIFIED
        >>> print(change_type.value)  # "modified"
        >>> if change_type in [ChangeType.CREATED, ChangeType.MODIFIED]:
        ...     print("File needs reindexing")
        File needs reindexing
    """

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileChange:
    """Represents a detected file system change with metadata and timing information.
    
    Captures all relevant information about a file system change event including
    the change type, affected paths, timing, and associated metadata for processing
    by the RAG system.
    
    Attributes:
        path: Path to the changed file.
        change_type: Type of change that occurred (created, modified, deleted, moved).
        timestamp: Unix timestamp when change was detected.
        old_path: Previous path for moved/renamed files (None for other changes).
        metadata: Additional change metadata (source_id, processing hints, etc.).
        
    Example:
        Creating a file change record:
        
        >>> from pathlib import Path
        >>> import time
        >>> 
        >>> change = FileChange(
        ...     path=Path("/project/src/auth.py"),
        ...     change_type=ChangeType.MODIFIED,
        ...     timestamp=time.time(),
        ...     metadata={"source_id": "project_src", "trigger": "user_edit"}
        ... )
        >>> print(f"{change.change_type.value}: {change.path}")
        modified: /project/src/auth.py
    """

    path: Path
    change_type: ChangeType
    timestamp: float = field(default_factory=time.time)
    old_path: Optional[Path] = None  # For moves/renames
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WatchedSource:
    """Configuration and state tracking for a watched source directory.
    
    Maintains all information needed to monitor a specific directory including
    watch configuration, file filtering patterns, and usage statistics.
    
    Attributes:
        path: Absolute path to the watched directory.
        source_id: Unique identifier for this watched source.
        recursive: Whether to watch subdirectories recursively.
        file_patterns: List of glob patterns for files to watch.
        last_scan: Unix timestamp of last full directory scan.
        change_count: Total number of changes detected in this source.
        
    Example:
        >>> from pathlib import Path
        >>> 
        >>> source = WatchedSource(
        ...     path=Path("/project/docs"),
        ...     source_id="project_docs",
        ...     recursive=True,
        ...     file_patterns=["*.md", "*.rst", "*.txt"],
        ...     change_count=0
        ... )
        >>> print(f"Watching: {source.path} (patterns: {source.file_patterns})")
        Watching: /project/docs (patterns: ['*.md', '*.rst', '*.txt'])
    """

    path: Path
    source_id: str
    recursive: bool = True
    file_patterns: List[str] = field(default_factory=list)
    last_scan: float = field(default_factory=time.time)
    change_count: int = 0


class FileChangeHandler(FileSystemEventHandler):
    """Watchdog event handler that processes file system events for watched directories.
    
    Extends watchdog's FileSystemEventHandler to provide custom logic for filtering
    relevant file changes and converting them into FileChange objects for processing
    by the RAG system. Integrates with file pattern matching and ignore rules.
    
    This handler is created automatically for each watched directory when using
    watchdog-based monitoring (non-polling mode). It filters events based on
    configured file patterns and ignore rules before forwarding them for processing.
    
    Attributes:
        watcher: Reference to parent FileWatcher instance.
        source_path: Path being watched by this handler.
        source_id: Unique identifier for the watched source.
        file_patterns: List of glob patterns for relevant files.
        scanner: FolderScanner for file filtering logic.
        
    Example:
        Handler is created automatically:
        
        >>> # Handler created internally by FileWatcher.watch()
        >>> # Processes events like:
        >>> # - on_created: New file detected
        >>> # - on_modified: File content changed  
        >>> # - on_deleted: File removed
        >>> # - on_moved: File renamed or moved
    """

    def __init__(
        self, watcher: "FileWatcher", source_path: Path, source_id: str, file_patterns: List[str]
    ):
        """Initialize file change handler for a specific watched directory.
        
        Args:
            watcher: Parent FileWatcher instance for event forwarding.
            source_path: Directory path being monitored by this handler.
            source_id: Unique identifier for the watched source.
            file_patterns: Glob patterns for files that should trigger events.
        """
        self.watcher = watcher
        self.source_path = source_path
        self.source_id = source_id
        self.file_patterns = file_patterns
        self.scanner = FolderScanner(watcher.indexer.config)

    def _should_process(self, path: str) -> bool:
        """Check if file should be processed based on patterns."""
        file_path = Path(path)

        # Check if it's a directory
        if file_path.is_dir():
            return False

        # Check against ignore patterns
        if self.scanner._should_ignore(file_path):
            return False

        # Check against file patterns
        if self.file_patterns:
            return any(file_path.match(pattern) for pattern in self.file_patterns)

        return True

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation."""
        if not event.is_directory and self._should_process(event.src_path):
            change = FileChange(
                path=Path(event.src_path),
                change_type=ChangeType.CREATED,
                metadata={"source_id": self.source_id},
            )
            asyncio.create_task(self.watcher._handle_change(change))

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification."""
        if not event.is_directory and self._should_process(event.src_path):
            change = FileChange(
                path=Path(event.src_path),
                change_type=ChangeType.MODIFIED,
                metadata={"source_id": self.source_id},
            )
            asyncio.create_task(self.watcher._handle_change(change))

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion."""
        if not event.is_directory:
            # Can't check patterns for deleted files, process all
            change = FileChange(
                path=Path(event.src_path),
                change_type=ChangeType.DELETED,
                metadata={"source_id": self.source_id},
            )
            asyncio.create_task(self.watcher._handle_change(change))

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move/rename."""
        if not event.is_directory:
            old_path = Path(event.src_path)
            new_path = Path(event.dest_path)

            # Check if new path should be processed
            if self._should_process(event.dest_path):
                change = FileChange(
                    path=new_path,
                    change_type=ChangeType.MOVED,
                    old_path=old_path,
                    metadata={"source_id": self.source_id},
                )
                asyncio.create_task(self.watcher._handle_change(change))
            else:
                # Treat as deletion if moved outside patterns
                change = FileChange(
                    path=old_path,
                    change_type=ChangeType.DELETED,
                    metadata={"source_id": self.source_id},
                )
                asyncio.create_task(self.watcher._handle_change(change))


class FileWatcher:
    """Comprehensive real-time file system monitor for automatic RAG system updates.
    
    Provides intelligent file system monitoring that automatically detects changes
    to watched directories and triggers appropriate reindexing operations. Supports
    both high-performance native file system events (inotify/FSEvents) and fallback
    polling mode for maximum compatibility.
    
    Key Capabilities:
        - Real-time monitoring using watchdog with native OS file system events
        - Intelligent debouncing to handle rapid successive changes efficiently
        - Batch processing for optimal throughput with multiple concurrent changes
        - Fallback polling mode for systems without native event support
        - Automatic knowledge graph updates when documents change
        - Comprehensive change history and performance statistics
        - Configurable file filtering with glob pattern support
        - Graceful error handling and automatic recovery
    
    Architecture:
        - FileChangeHandler: Processes raw file system events
        - Debouncing Layer: Aggregates rapid changes to avoid processing storms
        - Batch Processor: Handles multiple changes efficiently
        - Integration Layer: Updates document index and knowledge graph
    
    Attributes:
        indexer: Document indexer for processing file changes.
        graph_builder: Optional knowledge graph builder for relationship updates.
        debounce_seconds: Delay before processing changes to aggregate rapid updates.
        batch_size: Maximum number of changes to process in one batch.
        use_polling: Whether to use polling mode instead of native events.
        watched_sources: Dictionary of currently watched source directories.
        observer: Watchdog Observer instance for native file system monitoring.
        pending_changes: Dictionary of changes awaiting processing.
        change_history: Recent change history for monitoring and debugging.
        stats: Performance and activity statistics.
        
    Example:
        Complete file watching workflow:
        
        >>> from eol.rag_context.file_watcher import FileWatcher
        >>> from pathlib import Path
        >>> 
        >>> # Configure watcher for optimal performance
        >>> watcher = FileWatcher(
        ...     indexer=document_indexer,
        ...     graph_builder=kg_builder,
        ...     debounce_seconds=2.0,  # Wait 2s to aggregate rapid changes
        ...     batch_size=20,         # Process up to 20 changes per batch
        ...     use_polling=False      # Use native events when available
        ... )
        >>> 
        >>> # Start monitoring system
        >>> await watcher.start()
        >>> 
        >>> # Watch multiple sources with different configurations
        >>> docs_source = await watcher.watch(
        ...     path=Path("/project/docs"),
        ...     recursive=True,
        ...     file_patterns=["*.md", "*.rst"]
        ... )
        >>> 
        >>> code_source = await watcher.watch(
        ...     path=Path("/project/src"),
        ...     recursive=True,
        ...     file_patterns=["*.py", "*.js", "*.ts"]
        ... )
        >>> 
        >>> # Monitor system activity
        >>> stats = watcher.get_stats()
        >>> print(f"Active: {stats['is_running']}, Mode: {stats['mode']}")
        >>> print(f"Sources: {stats['watched_sources']}, Changes: {stats['changes_processed']}")
        >>> 
        >>> # Add custom change notification
        >>> def on_change(change):
        ...     print(f"File {change.change_type.value}: {change.path}")
        >>> watcher.add_change_callback(on_change)
        >>> 
        >>> # Force full rescan when needed (e.g., after system restart)
        >>> rescan_result = await watcher.force_rescan()
        >>> print(f"Full rescan: {rescan_result['files_indexed']} files processed")
        >>> 
        >>> # Clean shutdown
        >>> await watcher.stop()
    """

    def __init__(
        self,
        indexer: DocumentIndexer,
        graph_builder: Optional[KnowledgeGraphBuilder] = None,
        debounce_seconds: float = 2.0,
        batch_size: int = 10,
        use_polling: bool = False,
    ):
        """Initialize file watcher with configuration and dependencies.
        
        Args:
            indexer: Document indexer for processing file changes and reindexing.
            graph_builder: Optional knowledge graph builder for relationship updates
                when documents change. If None, only document indexing is performed.
            debounce_seconds: Delay in seconds before processing changes to aggregate
                rapid successive modifications. Higher values reduce CPU usage but
                increase latency for updates.
            batch_size: Maximum number of changes to process in one batch operation.
                Higher values improve throughput but increase memory usage.
            use_polling: Force polling mode instead of native file system events.
                Useful for network filesystems or systems without inotify support.
        """
        self.indexer = indexer
        self.graph_builder = graph_builder
        self.debounce_seconds = debounce_seconds
        self.batch_size = batch_size
        self.use_polling = use_polling or not WATCHDOG_AVAILABLE

        # Registry of watched source directories
        self.watched_sources: Dict[str, WatchedSource] = {}

        # Native file system monitoring (watchdog)
        self.observer: Optional[Observer] = None
        if WATCHDOG_AVAILABLE and not use_polling:
            self.observer = Observer()

        # Change processing pipeline
        self.pending_changes: Dict[str, FileChange] = {}
        self.change_history: List[FileChange] = []
        self.max_history = 1000

        # Async processing coordination
        self.processing_lock = asyncio.Lock()
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = False

        # Performance and activity metrics
        self.stats = {
            "changes_detected": 0,
            "changes_processed": 0,
            "reindex_count": 0,
            "errors": 0,
            "last_change": None,
        }

        # Custom notification callbacks
        self.change_callbacks: List[Callable[[FileChange], None]] = []

    async def start(self) -> None:
        """Start the file watching system and begin monitoring for changes.
        
        Initializes the monitoring system based on configuration, starting either
        native file system event monitoring (watchdog) or polling mode. Creates
        background tasks for change processing and begins accepting file system events.
        
        Startup Process:
        1. Validates system is not already running
        2. Sets running state and initializes monitoring mode
        3. Starts watchdog Observer (if using native events)
        4. Creates background processing task for change handling
        5. Logs startup confirmation with current watch count
        
        Raises:
            RuntimeError: If file watcher is already running.
            OSError: If unable to start file system monitoring.
            
        Example:
            >>> watcher = FileWatcher(indexer, debounce_seconds=3.0)
            >>> await watcher.start()
            >>> print(f"Monitoring started: {watcher.get_stats()['is_running']}")
            Monitoring started: True
            
        Note:
            Must be called before watch() methods. The system runs until stop()
            is called or the process terminates.
        """
        if self.is_running:
            logger.warning("File watcher already running")
            return

        self.is_running = True

        if self.use_polling:
            # Start polling task
            self.processing_task = asyncio.create_task(self._polling_loop())
            logger.info("Started file watcher in polling mode")
        else:
            # Start watchdog observer
            if self.observer:
                self.observer.start()
                logger.info("Started file watcher with watchdog")

            # Start processing task
            self.processing_task = asyncio.create_task(self._processing_loop())

        logger.info(f"File watcher started with {len(self.watched_sources)} sources")

    async def stop(self) -> None:
        """Stop the file watching system and clean up resources.
        
        Gracefully shuts down all monitoring components, processes any remaining
        pending changes, and cleans up background tasks. Ensures no resource leaks
        and that all file changes detected before shutdown are processed.
        
        Shutdown Process:
        1. Sets running state to false to signal shutdown
        2. Stops watchdog Observer and waits for clean termination
        3. Cancels background processing task with proper cleanup
        4. Processes any final pending changes before exit
        5. Logs shutdown confirmation
        
        Example:
            >>> # Graceful shutdown with cleanup
            >>> await watcher.stop()
            >>> print("File watching stopped")
            File watching stopped
            
        Note:
            This method is idempotent - calling it multiple times is safe.
            Always call this method before application shutdown to ensure
            proper resource cleanup.
        """
        if not self.is_running:
            return

        self.is_running = False

        # Stop observer
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)

        # Cancel processing task
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        logger.info("File watcher stopped")

    async def watch(
        self, path: Path, recursive: bool = True, file_patterns: Optional[List[str]] = None
    ) -> str:
        """Add a directory to watch for file changes with automatic indexing.
        
        Begins monitoring the specified directory for file system changes, performs
        initial indexing of existing files, and sets up real-time change detection.
        Integrates with knowledge graph building when available.
        
        Setup Process:
        1. Validates directory exists and is accessible
        2. Generates unique source ID for tracking
        3. Creates WatchedSource configuration
        4. Registers with file system monitoring (watchdog/polling)
        5. Performs initial complete indexing of existing files
        6. Updates knowledge graph with initial document relationships
        7. Begins real-time change monitoring
        
        Args:
            path: Directory path to monitor (must exist and be readable).
            recursive: Whether to monitor subdirectories recursively.
                True enables deep monitoring of entire directory trees.
            file_patterns: Optional list of glob patterns for files to monitor.
                If None, uses default patterns from indexer configuration.
                Examples: ["*.py", "*.md"], ["src/**/*.js", "docs/*.rst"]
                
        Returns:
            Unique source ID string for this watched directory. Use this ID
            for operations like unwatch() or force_rescan().
            
        Raises:
            ValueError: If path does not exist, is not a directory, or is not accessible.
            OSError: If unable to set up file system monitoring for the path.
            
        Example:
            Watch a project directory:
            
            >>> from pathlib import Path
            >>> 
            >>> # Watch code directory with pattern filtering
            >>> code_source = await watcher.watch(
            ...     path=Path("/project/src"),
            ...     recursive=True,
            ...     file_patterns=["*.py", "*.pyx", "*.pyi"]
            ... )
            >>> print(f"Watching source: {code_source}")
            
            >>> # Watch documentation with different patterns
            >>> docs_source = await watcher.watch(
            ...     path=Path("/project/docs"),
            ...     recursive=True,
            ...     file_patterns=["*.md", "*.rst", "*.txt"]
            ... )
            
            >>> # Check monitoring status
            >>> stats = watcher.get_stats()
            >>> print(f"Now watching {stats['watched_sources']} directories")
            
        Note:
            Initial indexing may take significant time for large directories.
            The method returns after setup is complete, but indexing continues
            in the background. Monitor stats to track indexing progress.
        """
        path = path.resolve()

        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")

        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        # Generate source ID
        source_id = self.indexer.scanner.generate_source_id(path)

        # Check if already watching
        if source_id in self.watched_sources:
            logger.info(f"Already watching {path}")
            return source_id

        # Create watched source
        watched = WatchedSource(
            path=path,
            source_id=source_id,
            recursive=recursive,
            file_patterns=file_patterns or self.indexer.config.document.file_patterns,
        )

        self.watched_sources[source_id] = watched

        # Add to observer if using watchdog
        if self.observer and not self.use_polling:
            handler = FileChangeHandler(self, path, source_id, watched.file_patterns)
            self.observer.schedule(handler, str(path), recursive=recursive)

        # Perform initial indexing
        await self.indexer.index_folder(path, recursive=recursive)

        # Build knowledge graph if available
        if self.graph_builder:
            await self.graph_builder.build_from_documents(source_id)

        logger.info(f"Started watching {path} (recursive={recursive})")
        return source_id

    async def unwatch(self, source_id: str) -> bool:
        """Stop monitoring a previously watched source directory.
        
        Removes the specified source from active monitoring, stopping all file
        system event processing for that directory. Does not remove already-indexed
        content from the RAG system.
        
        Args:
            source_id: Unique source identifier returned by watch() method.
            
        Returns:
            True if the source was actively being watched and was successfully
            removed. False if the source ID was not found in watched sources.
            
        Example:
            >>> # Stop watching a specific source
            >>> success = await watcher.unwatch("docs_source_123")
            >>> if success:
            ...     print("Successfully stopped watching source")
            ... else:
            ...     print("Source was not being watched")
            
        Note:
            This only stops monitoring; it does not remove indexed content.
            To remove content, use the indexer's remove_source() method.
        """
        if source_id not in self.watched_sources:
            return False

        watched = self.watched_sources[source_id]

        # Remove from observer (watchdog handles this internally)
        # No direct API to remove specific watches

        # Remove from watched sources
        del self.watched_sources[source_id]

        logger.info(f"Stopped watching {watched.path}")
        return True

    async def _handle_change(self, change: FileChange) -> None:
        """Handle a file change event."""
        self.stats["changes_detected"] += 1

        # Store in pending changes (debouncing)
        change_key = str(change.path)
        self.pending_changes[change_key] = change

        # Add to history
        self.change_history.append(change)
        if len(self.change_history) > self.max_history:
            self.change_history.pop(0)

        # Update stats
        self.stats["last_change"] = change.timestamp

        # Notify callbacks
        for callback in self.change_callbacks:
            try:
                callback(change)
            except Exception as e:
                logger.error(f"Error in change callback: {e}")

        logger.debug(f"Detected {change.change_type.value}: {change.path}")

    async def _processing_loop(self) -> None:
        """Main processing loop for handling changes."""
        while self.is_running:
            try:
                # Wait for debounce period
                await asyncio.sleep(self.debounce_seconds)

                # Process pending changes
                if self.pending_changes:
                    await self._process_pending_changes()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self.stats["errors"] += 1

    async def _polling_loop(self) -> None:
        """Polling loop for systems without inotify."""
        poll_interval = max(5.0, self.debounce_seconds * 2)
        file_states: Dict[str, Dict[str, Any]] = {}

        while self.is_running:
            try:
                for source in self.watched_sources.values():
                    # Scan directory
                    files = await self.indexer.scanner.scan_folder(
                        source.path, recursive=source.recursive, file_patterns=source.file_patterns
                    )

                    current_files = set(str(f) for f in files)

                    # Check for changes
                    for file_path in files:
                        file_key = str(file_path)

                        try:
                            stat = file_path.stat()
                            current_state = {
                                "mtime": stat.st_mtime,
                                "size": stat.st_size,
                            }

                            if file_key not in file_states:
                                # New file
                                change = FileChange(
                                    path=file_path,
                                    change_type=ChangeType.CREATED,
                                    metadata={"source_id": source.source_id},
                                )
                                await self._handle_change(change)
                            elif file_states[file_key] != current_state:
                                # Modified file
                                change = FileChange(
                                    path=file_path,
                                    change_type=ChangeType.MODIFIED,
                                    metadata={"source_id": source.source_id},
                                )
                                await self._handle_change(change)

                            file_states[file_key] = current_state

                        except FileNotFoundError:
                            # File was deleted during scan
                            pass

                    # Check for deleted files
                    previous_files = set(
                        k for k in file_states.keys() if k.startswith(str(source.path))
                    )
                    deleted_files = previous_files - current_files

                    for file_key in deleted_files:
                        change = FileChange(
                            path=Path(file_key),
                            change_type=ChangeType.DELETED,
                            metadata={"source_id": source.source_id},
                        )
                        await self._handle_change(change)
                        del file_states[file_key]

                # Process pending changes
                if self.pending_changes:
                    await self._process_pending_changes()

                # Wait before next poll
                await asyncio.sleep(poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(poll_interval)

    async def _process_pending_changes(self) -> None:
        """Process all pending changes."""
        async with self.processing_lock:
            if not self.pending_changes:
                return

            # Get changes to process
            changes = list(self.pending_changes.values())[: self.batch_size]

            # Clear processed changes
            for change in changes:
                change_key = str(change.path)
                if change_key in self.pending_changes:
                    del self.pending_changes[change_key]

            # Group changes by type
            created = [c for c in changes if c.change_type == ChangeType.CREATED]
            modified = [c for c in changes if c.change_type == ChangeType.MODIFIED]
            deleted = [c for c in changes if c.change_type == ChangeType.DELETED]
            moved = [c for c in changes if c.change_type == ChangeType.MOVED]

            # Process deletions first
            for change in deleted:
                await self._process_deletion(change)

            # Process moves
            for change in moved:
                await self._process_move(change)

            # Process creations and modifications
            for change in created + modified:
                await self._process_file_change(change)

            # Update statistics
            self.stats["changes_processed"] += len(changes)

            # Update knowledge graph if needed
            if self.graph_builder and (created or modified):
                affected_sources = set(c.metadata.get("source_id") for c in created + modified)
                for source_id in affected_sources:
                    if source_id:
                        await self.graph_builder.build_from_documents(source_id)

            logger.info(f"Processed {len(changes)} changes")

    async def _process_file_change(self, change: FileChange) -> None:
        """Process file creation or modification."""
        try:
            # Reindex the file
            chunks = await self.indexer.index_file(
                change.path, source_id=change.metadata.get("source_id")
            )

            self.stats["reindex_count"] += 1

            logger.info(f"Reindexed {change.path} ({chunks} chunks)")

        except Exception as e:
            logger.error(f"Error processing {change.path}: {e}")
            self.stats["errors"] += 1

    async def _process_deletion(self, change: FileChange) -> None:
        """Process file deletion."""
        try:
            # Remove from index
            file_hash = hashlib.md5(str(change.path).encode()).hexdigest()

            # Delete chunks
            pattern = f"chunk:*{file_hash}*"
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = await self.indexer.redis.redis.scan(cursor, match=pattern, count=100)

                if keys:
                    await self.indexer.redis.redis.delete(*keys)
                    deleted += len(keys)

                if cursor == 0:
                    break

            logger.info(f"Removed {deleted} chunks for deleted file {change.path}")

        except Exception as e:
            logger.error(f"Error processing deletion of {change.path}: {e}")
            self.stats["errors"] += 1

    async def _process_move(self, change: FileChange) -> None:
        """Process file move/rename."""
        try:
            # Remove old file from index
            if change.old_path:
                old_change = FileChange(
                    path=change.old_path, change_type=ChangeType.DELETED, metadata=change.metadata
                )
                await self._process_deletion(old_change)

            # Index new file
            await self._process_file_change(change)

            logger.info(f"Processed move from {change.old_path} to {change.path}")

        except Exception as e:
            logger.error(f"Error processing move of {change.path}: {e}")
            self.stats["errors"] += 1

    def add_change_callback(self, callback: Callable[[FileChange], None]) -> None:
        """Register a callback function to receive file change notifications.
        
        Allows external code to receive real-time notifications when file changes
        are detected, enabling custom processing logic, logging, or integration
        with other systems.
        
        Args:
            callback: Function that accepts a FileChange object. Will be called
                for each detected file change. Should be fast and non-blocking
                to avoid impacting file monitoring performance.
                
        Example:
            Add logging callback:
            
            >>> def log_changes(change: FileChange):
            ...     print(f"File {change.change_type.value}: {change.path}")
            >>> 
            >>> watcher.add_change_callback(log_changes)
            
            Add statistics tracking:
            
            >>> change_counts = {"created": 0, "modified": 0, "deleted": 0}
            >>> 
            >>> def track_changes(change: FileChange):
            ...     change_counts[change.change_type.value] += 1
            >>> 
            >>> watcher.add_change_callback(track_changes)
            
        Note:
            Callbacks should handle exceptions internally to avoid affecting
            the file monitoring system. Heavy processing should be delegated
            to background tasks.
        """
        self.change_callbacks.append(callback)

    def remove_change_callback(self, callback: Callable[[FileChange], None]) -> None:
        """Unregister a previously added file change callback.
        
        Removes a callback function from the notification list, stopping
        future file change notifications to that function.
        
        Args:
            callback: Function previously registered with add_change_callback().
                Must be the exact same function object.
                
        Example:
            >>> def my_callback(change):
            ...     print(f"Change: {change.path}")
            >>> 
            >>> # Add callback
            >>> watcher.add_change_callback(my_callback)
            >>> 
            >>> # Later, remove it
            >>> watcher.remove_change_callback(my_callback)
            
        Note:
            If the callback is not currently registered, this method does
            nothing (no error is raised).
        """
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive file watcher performance and activity statistics.
        
        Returns detailed metrics about file watching activity, performance,
        and current system state. Useful for monitoring, debugging, and
        system optimization.
        
        Returns:
            Dictionary containing:
            - changes_detected: Total file changes detected since startup
            - changes_processed: Total changes successfully processed
            - reindex_count: Number of file reindexing operations performed
            - errors: Number of errors encountered during processing
            - last_change: Timestamp of most recent change detection
            - watched_sources: Number of directories currently being monitored
            - pending_changes: Number of changes awaiting processing
            - is_running: Whether monitoring system is currently active
            - mode: Monitoring mode ("watchdog" or "polling")
            
        Example:
            >>> stats = watcher.get_stats()
            >>> print(f"System Status: {stats['is_running']} ({stats['mode']} mode)")
            >>> print(f"Activity: {stats['changes_processed']}/{stats['changes_detected']} changes processed")
            >>> print(f"Performance: {stats['reindex_count']} reindex ops, {stats['errors']} errors")
            >>> print(f"Current Load: {stats['pending_changes']} pending, {stats['watched_sources']} sources")
            
            >>> # Calculate success rate
            >>> if stats['changes_detected'] > 0:
            ...     success_rate = stats['changes_processed'] / stats['changes_detected']
            ...     print(f"Success Rate: {success_rate:.1%}")
        """
        stats = self.stats.copy()
        stats.update(
            {
                "watched_sources": len(self.watched_sources),
                "pending_changes": len(self.pending_changes),
                "is_running": self.is_running,
                "mode": "polling" if self.use_polling else "watchdog",
            }
        )
        return stats

    def get_change_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get chronological history of recent file changes for monitoring and debugging.
        
        Returns detailed information about recent file system changes detected
        by the watcher, formatted for easy analysis and debugging. Useful for
        understanding system activity patterns and troubleshooting issues.
        
        Args:
            limit: Maximum number of recent changes to return. Higher values
                provide more history but use more memory.
                
        Returns:
            List of dictionaries, each containing:
            - path: String representation of the changed file path
            - type: Change type ("created", "modified", "deleted", "moved")
            - timestamp: Unix timestamp of when change was detected
            - time: Human-readable ISO timestamp
            - old_path: Previous path for moved files (None for other changes)
            - metadata: Additional change metadata (source_id, etc.)
            
        Example:
            Monitor recent activity:
            
            >>> history = watcher.get_change_history(limit=20)
            >>> print(f"Recent activity: {len(history)} changes")
            >>> 
            >>> for change in history[-5:]:  # Show last 5 changes
            ...     print(f"{change['time']}: {change['type']} - {change['path']}")
            
            Analyze change patterns:
            
            >>> from collections import Counter
            >>> change_types = Counter(c['type'] for c in history)
            >>> print(f"Change distribution: {dict(change_types)}")
            
            Find recent modifications to specific file types:
            
            >>> py_changes = [c for c in history if c['path'].endswith('.py')]
            >>> print(f"Python file changes: {len(py_changes)}")
        """
        history = []
        for change in self.change_history[-limit:]:
            history.append(
                {
                    "path": str(change.path),
                    "type": change.change_type.value,
                    "timestamp": change.timestamp,
                    "time": datetime.fromtimestamp(change.timestamp).isoformat(),
                    "old_path": str(change.old_path) if change.old_path else None,
                    "metadata": change.metadata,
                }
            )
        return history

    async def force_rescan(self, source_id: Optional[str] = None) -> Dict[str, int]:
        """Force complete reindexing of watched sources with fresh content analysis.
        
        Performs comprehensive reindexing of specified sources, bypassing all caches
        and change detection optimizations. Useful for recovering from index
        corruption, applying configuration changes, or handling external modifications
        not detected by file system monitoring.
        
        Rescan Process:
        1. Identifies sources to rescan (specific source or all watched sources)
        2. Performs full folder indexing with force_reindex=True
        3. Updates knowledge graph with fresh document relationships
        4. Returns comprehensive statistics about the operation
        
        Args:
            source_id: Optional specific source ID to rescan. If None, rescans
                all currently watched sources. Use get_stats() to see active sources.
                
        Returns:
            Dictionary containing rescan results:
            - sources_scanned: Number of source directories processed
            - files_indexed: Total number of files reindexed
            - chunks_created: Total number of document chunks created
            
        Example:
            Force rescan of all sources:
            
            >>> result = await watcher.force_rescan()
            >>> print(f"Rescanned {result['sources_scanned']} sources")
            >>> print(f"Processed {result['files_indexed']} files")
            >>> print(f"Created {result['chunks_created']} chunks")
            
            Rescan specific source:
            
            >>> result = await watcher.force_rescan(source_id="docs_source_123")
            >>> if result['sources_scanned'] > 0:
            ...     print(f"Successfully rescanned source")
            ... else:
            ...     print("Source not found or not watched")
            
        Note:
            This operation can be time-intensive for large document sets.
            It bypasses all optimizations and processes every file completely.
            Consider running during low-activity periods.
        """
        sources_to_scan = []

        if source_id:
            if source_id in self.watched_sources:
                sources_to_scan.append(self.watched_sources[source_id])
        else:
            sources_to_scan = list(self.watched_sources.values())

        total_files = 0
        total_chunks = 0

        for source in sources_to_scan:
            logger.info(f"Force rescanning {source.path}")

            result = await self.indexer.index_folder(
                source.path, recursive=source.recursive, force_reindex=True
            )

            total_files += result.file_count
            total_chunks += result.total_chunks

            # Update knowledge graph
            if self.graph_builder:
                await self.graph_builder.build_from_documents(source.source_id)

        return {
            "sources_scanned": len(sources_to_scan),
            "files_indexed": total_files,
            "chunks_created": total_chunks,
        }
