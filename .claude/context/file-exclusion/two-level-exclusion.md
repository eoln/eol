# Two-Level Exclusion Strategy for File Indexing

## Critical Insight

Exclusion must happen at **TWO LEVELS** to ensure complete protection against indexing forbidden resources:

1. **LEVEL 1: Folder Scanning** - When building the initial list of folders to process
2. **LEVEL 2: File Iteration** - When iterating through files within each folder

## Current Problem Areas

### Level 1: Folder List Creation

```python
# Current issue - may include .venv in folder list
folders_to_process = get_all_folders(root_path)  # This might include .venv!
for folder in folders_to_process:
    process_folder(folder)  # Too late to exclude!
```

### Level 2: File Scanning Within Folders

```python
# Current issue - rglob traverses into .venv
for pattern in file_patterns:
    paths = folder_path.rglob(pattern)  # Traverses ALL subdirs including .venv
    for path in paths:
        # Post-traversal filtering is unreliable
        if should_skip:
            continue
```

## Complete Two-Level Solution

### Level 1: Folder-Level Exclusion

```python
class FolderScanner:
    """Scanner with two-level exclusion strategy"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.excluded_dirs = {
            ".venv", "venv", ".env", "env",
            ".uv-cache", "__pycache__", ".pytest_cache",
            "node_modules", ".npm", ".yarn",
            "dist", "build", "target",
            ".git", ".svn", ".hg",
            ".idea", ".vscode",
            "coverage", ".coverage", "htmlcov"
        }

    async def get_folders_to_index(
        self,
        root_path: Path,
        recursive: bool = True
    ) -> List[Path]:
        """Level 1: Get list of folders to process, excluding forbidden ones"""
        folders = [root_path]  # Always include root

        if recursive:
            folders.extend(self._get_valid_subdirectories(root_path))

        return folders

    def _get_valid_subdirectories(self, root: Path) -> List[Path]:
        """Recursively get subdirectories, excluding forbidden ones"""
        valid_dirs = []

        def walk_dirs(directory: Path):
            try:
                for item in directory.iterdir():
                    if item.is_dir():
                        # LEVEL 1 EXCLUSION: Skip forbidden directories
                        if self._should_exclude_directory(item):
                            logger.debug(f"Level 1 exclusion: Skipping directory {item}")
                            continue

                        valid_dirs.append(item)
                        # Recursively get subdirectories
                        walk_dirs(item)
            except PermissionError:
                logger.warning(f"Permission denied: {directory}")

        walk_dirs(root)
        return valid_dirs

    def _should_exclude_directory(self, dir_path: Path) -> bool:
        """Check if directory should be excluded at Level 1"""
        # Check directory name
        if dir_path.name in self.excluded_dirs:
            return True

        # Check if it's a hidden directory (except allowed ones)
        if dir_path.name.startswith('.'):
            allowed_hidden = {'.github', '.claude'}
            return dir_path.name not in allowed_hidden

        # Check parent directories
        for parent in dir_path.parents:
            if parent.name in self.excluded_dirs:
                return True

        return False
```

### Level 2: File-Level Exclusion

```python
    async def scan_files_in_folder(
        self,
        folder_path: Path,
        file_patterns: List[str],
        recursive: bool = False  # Already handled at Level 1
    ) -> List[Path]:
        """Level 2: Scan files within a specific folder with exclusion"""
        files_to_index = []

        # Extract extensions for faster matching
        extensions = set()
        for pattern in file_patterns:
            if pattern.startswith("*."):
                extensions.add(pattern[1:])

        # Use iterdir for current folder only (recursion handled at Level 1)
        try:
            for item in folder_path.iterdir():
                if item.is_file():
                    # Check file patterns
                    if item.suffix in extensions:
                        if not self._should_exclude_file(item):
                            files_to_index.append(item)

                elif item.is_dir() and recursive:
                    # LEVEL 2 EXCLUSION: Double-check directories
                    if self._should_exclude_directory(item):
                        logger.debug(f"Level 2 exclusion: Skipping {item}")
                        continue

                    # Recursively scan subdirectory
                    subdir_files = await self.scan_files_in_folder(
                        item, file_patterns, recursive=True
                    )
                    files_to_index.extend(subdir_files)

        except PermissionError:
            logger.warning(f"Permission denied: {folder_path}")

        return files_to_index

    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded at Level 2"""
        # Check file patterns
        exclude_patterns = {
            "*.pyc", "*.pyo", "*.pyd",
            ".DS_Store", "Thumbs.db",
            "*.log", "*.tmp", "*.temp",
            "*.lock", "*.pid"
        }

        for pattern in exclude_patterns:
            if file_path.match(pattern):
                return True

        # Check if file is in an excluded directory (safety check)
        for parent in file_path.parents:
            if parent.name in self.excluded_dirs:
                logger.warning(f"File {file_path} in excluded dir {parent} - should not happen!")
                return True

        return False
```

### Complete Integration

```python
class DocumentIndexer:
    """Indexer with two-level exclusion"""

    async def index_folder(
        self,
        folder_path: Path,
        source_id: str = None,
        recursive: bool = True,
        force_reindex: bool = False
    ) -> IndexedSource:
        """Index folder with two-level exclusion"""

        scanner = FolderScanner(self.config)

        # LEVEL 1: Get valid folders to process
        folders_to_process = await scanner.get_folders_to_index(
            folder_path,
            recursive=recursive
        )

        logger.info(f"Level 1: Found {len(folders_to_process)} folders to process")

        all_files = []

        # Process each valid folder
        for folder in folders_to_process:
            # LEVEL 2: Scan files within each folder
            files = await scanner.scan_files_in_folder(
                folder,
                self.config.document.file_patterns,
                recursive=False  # Don't recurse, we already have folder list
            )

            logger.debug(f"Level 2: Found {len(files)} files in {folder}")
            all_files.extend(files)

        logger.info(f"Total files to index: {len(all_files)}")

        # Index files
        for file_path in all_files:
            await self.index_file(file_path, source_id)

        return IndexedSource(...)
```

## Alternative: Unified Two-Level Approach

```python
class OptimizedScanner:
    """Unified scanner with built-in two-level exclusion"""

    async def scan_folder_complete(
        self,
        root_path: Path,
        file_patterns: List[str],
        recursive: bool = True
    ) -> List[Path]:
        """Complete scan with two-level exclusion built-in"""

        files_to_index = []
        processed_dirs = set()

        def should_process_dir(dir_path: Path) -> bool:
            """Unified directory exclusion check"""
            # Already processed?
            if dir_path in processed_dirs:
                return False

            # Name-based exclusion
            if dir_path.name in self.excluded_dirs:
                return False

            # Hidden directory exclusion
            if dir_path.name.startswith('.'):
                return dir_path.name in {'.github', '.claude'}

            return True

        def scan_directory_safe(directory: Path, depth: int = 0):
            """Scan with both levels of protection"""

            # LEVEL 1: Check if we should process this directory
            if not should_process_dir(directory):
                logger.debug(f"Skipping directory: {directory}")
                return

            processed_dirs.add(directory)

            try:
                for item in directory.iterdir():
                    if item.is_file():
                        # LEVEL 2: File-level checks
                        if self._matches_patterns(item, file_patterns):
                            if not self._should_exclude_file(item):
                                files_to_index.append(item)

                    elif item.is_dir() and (recursive or depth == 0):
                        # Recursive scan with both levels
                        scan_directory_safe(item, depth + 1)

            except PermissionError:
                logger.warning(f"Permission denied: {directory}")

        # Start scanning
        scan_directory_safe(root_path)

        return sorted(set(files_to_index))
```

## Testing Two-Level Exclusion

```python
@pytest.mark.asyncio
async def test_two_level_exclusion():
    """Test that exclusion works at both levels"""

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create test structure
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("# main")

        # Create .venv with nested structure
        venv = root / ".venv"
        (venv / "lib" / "python3.9" / "site-packages").mkdir(parents=True)
        (venv / "lib" / "python3.9" / "site-packages" / "test.py").write_text("# should not index")

        # Create .uv-cache
        cache = root / ".uv-cache"
        (cache / "packages").mkdir(parents=True)
        (cache / "packages" / "data.py").write_text("# should not index")

        scanner = FolderScanner(config)

        # Test Level 1: Folder list should not include .venv
        folders = await scanner.get_folders_to_index(root, recursive=True)
        folder_names = [f.name for f in folders]

        assert "src" in folder_names
        assert ".venv" not in folder_names
        assert ".uv-cache" not in folder_names

        # Test Level 2: File scan should not find files in .venv
        all_files = []
        for folder in folders:
            files = await scanner.scan_files_in_folder(
                folder, ["*.py"], recursive=False
            )
            all_files.extend(files)

        file_paths = [str(f.relative_to(root)) for f in all_files]

        assert "src/main.py" in file_paths
        assert not any(".venv" in path for path in file_paths)
        assert not any(".uv-cache" in path for path in file_paths)

@pytest.mark.asyncio
async def test_exclusion_with_deep_nesting():
    """Test exclusion with deeply nested forbidden directories"""

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create valid structure
        (root / "project" / "src" / "core").mkdir(parents=True)
        (root / "project" / "src" / "core" / "app.py").write_text("# app")

        # Create forbidden nested structure
        (root / "project" / ".venv" / "lib" / "site-packages").mkdir(parents=True)
        (root / "project" / ".venv" / "lib" / "site-packages" / "bad.py").write_text("# bad")

        # Create another forbidden structure
        (root / "project" / "node_modules" / "package" / "dist").mkdir(parents=True)
        (root / "project" / "node_modules" / "package" / "dist" / "bad.js").write_text("// bad")

        scanner = OptimizedScanner(config)
        files = await scanner.scan_folder_complete(root, ["*.py", "*.js"], recursive=True)

        file_paths = [str(f.relative_to(root)) for f in files]

        # Should only find the valid file
        assert len(files) == 1
        assert "project/src/core/app.py" in file_paths

        # Should not find forbidden files
        assert not any(".venv" in path for path in file_paths)
        assert not any("node_modules" in path for path in file_paths)
```

## Performance Metrics

```python
@dataclass
class ScanMetrics:
    """Metrics for two-level scanning"""

    # Level 1 metrics
    total_dirs_found: int
    dirs_excluded_level1: int
    dirs_processed: int

    # Level 2 metrics
    total_files_found: int
    files_excluded_level2: int
    files_indexed: int

    # Performance
    scan_time_seconds: float

    @property
    def level1_exclusion_rate(self) -> float:
        if self.total_dirs_found == 0:
            return 0
        return self.dirs_excluded_level1 / self.total_dirs_found

    @property
    def level2_exclusion_rate(self) -> float:
        if self.total_files_found == 0:
            return 0
        return self.files_excluded_level2 / self.total_files_found

    def log_summary(self):
        logger.info(f"""
        Scan Metrics:
        - Level 1: {self.dirs_excluded_level1}/{self.total_dirs_found} dirs excluded ({self.level1_exclusion_rate:.1%})
        - Level 2: {self.files_excluded_level2}/{self.total_files_found} files excluded ({self.level2_exclusion_rate:.1%})
        - Time: {self.scan_time_seconds:.2f}s
        - Result: {self.files_indexed} files ready for indexing
        """)
```

## Implementation Checklist

### Level 1: Folder Exclusion

- [ ] Create `get_folders_to_index()` method
- [ ] Implement directory tree walker with exclusion
- [ ] Add metrics for excluded directories
- [ ] Test with nested forbidden directories

### Level 2: File Exclusion

- [ ] Create `scan_files_in_folder()` method
- [ ] Add file-level exclusion patterns
- [ ] Implement safety check for parent directories
- [ ] Add metrics for excluded files

### Integration

- [ ] Update `index_folder()` to use two-level approach
- [ ] Add comprehensive logging at both levels
- [ ] Create performance benchmarks
- [ ] Write integration tests

### Validation

- [ ] Test with real .venv (10k+ files)
- [ ] Test with node_modules (1000s of nested dirs)
- [ ] Verify zero leakage of forbidden files
- [ ] Confirm performance improvement

## Key Benefits

1. **Complete Protection**: No forbidden files can slip through
2. **Performance**: Avoid traversing large forbidden directories
3. **Clarity**: Clear separation of concerns at each level
4. **Debugging**: Easy to track where exclusions happen
5. **Flexibility**: Can customize exclusions at each level

This two-level approach ensures absolute protection against indexing forbidden resources while maintaining optimal performance.
