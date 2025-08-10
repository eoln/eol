"""
File watcher for real-time source change tracking and propagation.
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
    """Types of file system changes."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileChange:
    """Represents a file system change."""
    path: Path
    change_type: ChangeType
    timestamp: float = field(default_factory=time.time)
    old_path: Optional[Path] = None  # For moves/renames
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WatchedSource:
    """Represents a watched source directory."""
    path: Path
    source_id: str
    recursive: bool = True
    file_patterns: List[str] = field(default_factory=list)
    last_scan: float = field(default_factory=time.time)
    change_count: int = 0


class FileChangeHandler(FileSystemEventHandler):
    """Handles file system events for watched directories."""
    
    def __init__(
        self,
        watcher: 'FileWatcher',
        source_path: Path,
        source_id: str,
        file_patterns: List[str]
    ):
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
                metadata={"source_id": self.source_id}
            )
            asyncio.create_task(self.watcher._handle_change(change))
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification."""
        if not event.is_directory and self._should_process(event.src_path):
            change = FileChange(
                path=Path(event.src_path),
                change_type=ChangeType.MODIFIED,
                metadata={"source_id": self.source_id}
            )
            asyncio.create_task(self.watcher._handle_change(change))
    
    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion."""
        if not event.is_directory:
            # Can't check patterns for deleted files, process all
            change = FileChange(
                path=Path(event.src_path),
                change_type=ChangeType.DELETED,
                metadata={"source_id": self.source_id}
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
                    metadata={"source_id": self.source_id}
                )
                asyncio.create_task(self.watcher._handle_change(change))
            else:
                # Treat as deletion if moved outside patterns
                change = FileChange(
                    path=old_path,
                    change_type=ChangeType.DELETED,
                    metadata={"source_id": self.source_id}
                )
                asyncio.create_task(self.watcher._handle_change(change))


class FileWatcher:
    """
    Watches file system for changes and updates RAG index in real-time.
    
    Features:
    - Real-time file monitoring using watchdog
    - Debouncing for rapid changes
    - Batch processing for efficiency
    - Fallback polling for systems without inotify
    - Change history tracking
    """
    
    def __init__(
        self,
        indexer: DocumentIndexer,
        graph_builder: Optional[KnowledgeGraphBuilder] = None,
        debounce_seconds: float = 2.0,
        batch_size: int = 10,
        use_polling: bool = False
    ):
        self.indexer = indexer
        self.graph_builder = graph_builder
        self.debounce_seconds = debounce_seconds
        self.batch_size = batch_size
        self.use_polling = use_polling or not WATCHDOG_AVAILABLE
        
        # Watched sources
        self.watched_sources: Dict[str, WatchedSource] = {}
        
        # File system observer
        self.observer: Optional[Observer] = None
        if WATCHDOG_AVAILABLE and not use_polling:
            self.observer = Observer()
        
        # Change tracking
        self.pending_changes: Dict[str, FileChange] = {}
        self.change_history: List[FileChange] = []
        self.max_history = 1000
        
        # Processing state
        self.processing_lock = asyncio.Lock()
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            "changes_detected": 0,
            "changes_processed": 0,
            "reindex_count": 0,
            "errors": 0,
            "last_change": None,
        }
        
        # Callbacks
        self.change_callbacks: List[Callable[[FileChange], None]] = []
    
    async def start(self) -> None:
        """Start watching for file changes."""
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
        """Stop watching for file changes."""
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
        self,
        path: Path,
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None
    ) -> str:
        """
        Add a directory to watch for changes.
        
        Args:
            path: Directory path to watch
            recursive: Watch subdirectories
            file_patterns: File patterns to watch
            
        Returns:
            Source ID for the watched directory
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
            file_patterns=file_patterns or self.indexer.config.document.file_patterns
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
        """
        Stop watching a source directory.
        
        Args:
            source_id: Source ID to stop watching
            
        Returns:
            True if source was being watched
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
                        source.path,
                        recursive=source.recursive,
                        file_patterns=source.file_patterns
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
                                    metadata={"source_id": source.source_id}
                                )
                                await self._handle_change(change)
                            elif file_states[file_key] != current_state:
                                # Modified file
                                change = FileChange(
                                    path=file_path,
                                    change_type=ChangeType.MODIFIED,
                                    metadata={"source_id": source.source_id}
                                )
                                await self._handle_change(change)
                            
                            file_states[file_key] = current_state
                            
                        except FileNotFoundError:
                            # File was deleted during scan
                            pass
                    
                    # Check for deleted files
                    previous_files = set(
                        k for k in file_states.keys()
                        if k.startswith(str(source.path))
                    )
                    deleted_files = previous_files - current_files
                    
                    for file_key in deleted_files:
                        change = FileChange(
                            path=Path(file_key),
                            change_type=ChangeType.DELETED,
                            metadata={"source_id": source.source_id}
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
            changes = list(self.pending_changes.values())[:self.batch_size]
            
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
                change.path,
                source_id=change.metadata.get("source_id")
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
                cursor, keys = await self.indexer.redis.redis.scan(
                    cursor,
                    match=pattern,
                    count=100
                )
                
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
                    path=change.old_path,
                    change_type=ChangeType.DELETED,
                    metadata=change.metadata
                )
                await self._process_deletion(old_change)
            
            # Index new file
            await self._process_file_change(change)
            
            logger.info(f"Processed move from {change.old_path} to {change.path}")
            
        except Exception as e:
            logger.error(f"Error processing move of {change.path}: {e}")
            self.stats["errors"] += 1
    
    def add_change_callback(self, callback: Callable[[FileChange], None]) -> None:
        """Add a callback for file changes."""
        self.change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[FileChange], None]) -> None:
        """Remove a change callback."""
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get watcher statistics."""
        stats = self.stats.copy()
        stats.update({
            "watched_sources": len(self.watched_sources),
            "pending_changes": len(self.pending_changes),
            "is_running": self.is_running,
            "mode": "polling" if self.use_polling else "watchdog",
        })
        return stats
    
    def get_change_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent change history."""
        history = []
        for change in self.change_history[-limit:]:
            history.append({
                "path": str(change.path),
                "type": change.change_type.value,
                "timestamp": change.timestamp,
                "time": datetime.fromtimestamp(change.timestamp).isoformat(),
                "old_path": str(change.old_path) if change.old_path else None,
                "metadata": change.metadata,
            })
        return history
    
    async def force_rescan(self, source_id: Optional[str] = None) -> Dict[str, int]:
        """
        Force a full rescan of watched sources.
        
        Args:
            source_id: Specific source to rescan, or None for all
            
        Returns:
            Dict with rescan statistics
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
                source.path,
                recursive=source.recursive,
                force_reindex=True
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