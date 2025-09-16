#!/usr/bin/env python
"""
Fix for file exclusion issue - implements two-level exclusion strategy.
This script shows the complete implementation that should replace the current
broken scan_folder method.
"""

import logging
from pathlib import Path
from typing import List, Set

logger = logging.getLogger(__name__)


class ImprovedFolderScanner:
    """Folder scanner with proper two-level exclusion strategy"""

    def __init__(self, config=None):
        """Initialize with excluded directories list"""
        self.config = config

        # Comprehensive list of directories to exclude
        self.excluded_dirs: Set[str] = {
            # Virtual environments
            ".venv",
            "venv",
            ".env",
            "env",
            "virtualenv",
            ".virtualenv",
            # Package managers and caches
            ".uv-cache",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".npm",
            ".yarn",
            ".pnpm-store",
            ".pip-cache",
            ".pip",
            ".eggs",
            "*.egg-info",
            # Build and distribution
            "dist",
            "build",
            "target",
            "_build",
            "out",
            "output",
            ".next",
            ".nuxt",
            # Version control
            ".git",
            ".svn",
            ".hg",
            ".bzr",
            # IDE and editors
            ".idea",
            ".vscode",
            ".vs",
            ".sublime",
            ".atom",
            ".eclipse",
            ".netbeans",
            # Testing and coverage
            "coverage",
            ".coverage",
            "htmlcov",
            ".tox",
            ".nox",
            ".hypothesis",
            # Language specific
            ".cargo",
            ".rustup",
            ".gradle",
            ".m2",
            ".stack",
            ".cabal",
            ".gem",
            ".bundle",
            # OS specific
            ".DS_Store",
            "Thumbs.db",
            "$RECYCLE.BIN",
            # Documentation builds
            "_site",
            "site",
            ".docusaurus",
            # Temporary
            "tmp",
            "temp",
            ".tmp",
            ".temp",
            ".cache",
        }

        # File patterns to exclude
        self.excluded_file_patterns: Set[str] = {
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "*.so",
            "*.dylib",
            "*.dll",
            "*.class",
            "*.jar",
            "*.log",
            "*.pid",
            "*.lock",
            "*.swp",
            "*.swo",
            "*~",
            ".DS_Store",
            "Thumbs.db",
            "*.tmp",
            "*.temp",
            "*.bak",
        }

    async def scan_folder(
        self,
        folder_path: Path | str,
        recursive: bool = True,
        respect_gitignore: bool = True,
        file_patterns: List[str] | None = None,
    ) -> List[Path]:
        """
        Scan folder with TWO-LEVEL exclusion strategy.

        Level 1: Exclude directories from traversal
        Level 2: Exclude files during iteration
        """
        if isinstance(folder_path, str):
            folder_path = Path(folder_path)

        if not folder_path.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")

        if not folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")

        folder_path = folder_path.resolve()
        file_patterns = file_patterns or ["*.py", "*.md", "*.txt", "*.json", "*.yaml", "*.yml"]

        # Load gitignore if requested
        gitignore_matcher = None
        if respect_gitignore:
            try:
                import gitignore_parser

                gitignore_path = folder_path / ".gitignore"
                if gitignore_path.exists():
                    gitignore_matcher = gitignore_parser.parse_gitignore(gitignore_path)
            except ImportError:
                logger.warning("gitignore_parser not available")

        # Use two-level scanning
        files_to_index = await self._scan_with_two_level_exclusion(
            folder_path, file_patterns, recursive, gitignore_matcher
        )

        logger.info(f"Found {len(files_to_index)} files to index in {folder_path}")
        return files_to_index

    async def _scan_with_two_level_exclusion(
        self, root_path: Path, file_patterns: List[str], recursive: bool, gitignore_matcher
    ) -> List[Path]:
        """Implement two-level exclusion strategy"""

        files_to_index = []
        dirs_excluded = 0
        files_excluded = 0

        # Extract extensions for faster matching
        extensions = set()
        complex_patterns = []

        for pattern in file_patterns:
            if pattern.startswith("*.") and "*" not in pattern[2:]:
                extensions.add(pattern[1:])  # .py, .md, etc.
            else:
                complex_patterns.append(pattern)

        def should_exclude_directory(dir_path: Path) -> bool:
            """LEVEL 1: Check if directory should be excluded"""

            # Check against excluded directory names
            if dir_path.name in self.excluded_dirs:
                return True

            # Check if any parent is excluded (shouldn't happen with proper traversal)
            for parent in dir_path.parents:
                if parent.name in self.excluded_dirs:
                    logger.warning(f"Directory {dir_path} has excluded parent {parent}")
                    return True

            # Check gitignore
            if gitignore_matcher:
                try:
                    if gitignore_matcher(str(dir_path.relative_to(root_path))):
                        return True
                except (ValueError, OSError):
                    pass

            # Exclude hidden directories except specific ones
            if dir_path.name.startswith("."):
                allowed_hidden = {".github", ".claude", ".config"}
                return dir_path.name not in allowed_hidden

            return False

        def should_exclude_file(file_path: Path) -> bool:
            """LEVEL 2: Check if file should be excluded"""

            # Check against excluded file patterns
            for pattern in self.excluded_file_patterns:
                if file_path.match(pattern):
                    return True

            # Check gitignore
            if gitignore_matcher:
                try:
                    if gitignore_matcher(str(file_path.relative_to(root_path))):
                        return True
                except (ValueError, OSError):
                    pass

            # Safety check: ensure file is not in excluded directory
            for parent in file_path.parents:
                if parent.name in self.excluded_dirs:
                    logger.error(
                        f"File {file_path} is in excluded directory {parent} - "
                        "this should not happen!"
                    )
                    return True

            return False

        def scan_directory(directory: Path, depth: int = 0):
            """Recursively scan directory with two-level exclusion"""

            nonlocal dirs_excluded, files_excluded

            try:
                for entry in sorted(directory.iterdir()):
                    if entry.is_dir():
                        # LEVEL 1 EXCLUSION: Check directory before traversing
                        if should_exclude_directory(entry):
                            logger.debug(
                                f"Level 1: Excluding directory {entry.relative_to(root_path)}"
                            )
                            dirs_excluded += 1
                            continue  # Don't traverse into this directory at all

                        # Only recurse if recursive mode or at root level
                        if recursive or depth == 0:
                            scan_directory(entry, depth + 1)

                    elif entry.is_file():
                        # Check if file matches our patterns
                        matches_pattern = False

                        # Quick check by extension
                        if entry.suffix in extensions:
                            matches_pattern = True
                        # Check complex patterns
                        elif any(entry.match(p) for p in complex_patterns):
                            matches_pattern = True

                        if matches_pattern:
                            # LEVEL 2 EXCLUSION: Check file before adding
                            if should_exclude_file(entry):
                                logger.debug(
                                    f"Level 2: Excluding file {entry.relative_to(root_path)}"
                                )
                                files_excluded += 1
                            else:
                                files_to_index.append(entry)

            except PermissionError:
                logger.warning(f"Permission denied accessing: {directory}")
            except Exception as e:
                logger.error(f"Error scanning directory {directory}: {e}")

        # Start scanning from root
        scan_directory(root_path)

        # Log exclusion statistics
        logger.info(
            f"Exclusion stats: {dirs_excluded} directories excluded, "
            f"{files_excluded} files excluded"
        )

        # Remove duplicates and sort
        return sorted(set(files_to_index))


# Test the implementation
async def test_exclusion():
    """Test that the two-level exclusion works correctly"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create test structure
        print(f"Creating test structure in {root}")

        # Valid files
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("print('main')")
        (root / "src" / "utils.py").write_text("# utils")

        (root / "docs").mkdir()
        (root / "docs" / "README.md").write_text("# Documentation")

        # Create .venv with many files (should be excluded)
        venv = root / ".venv"
        (venv / "lib" / "python3.9" / "site-packages").mkdir(parents=True)
        for i in range(100):
            (venv / "lib" / "python3.9" / "site-packages" / f"package_{i}.py").write_text(
                f"# package {i}"
            )

        # Create .uv-cache (should be excluded)
        cache = root / ".uv-cache"
        (cache / "packages").mkdir(parents=True)
        for i in range(50):
            (cache / "packages" / f"cache_{i}.py").write_text(f"# cache {i}")

        # Create node_modules (should be excluded)
        node = root / "node_modules"
        (node / "package1" / "dist").mkdir(parents=True)
        (node / "package1" / "dist" / "index.js").write_text("// javascript")

        # Create __pycache__ (should be excluded)
        pycache = root / "src" / "__pycache__"
        pycache.mkdir()
        (pycache / "main.cpython-39.pyc").write_text("# bytecode")

        # Test the scanner
        scanner = ImprovedFolderScanner()
        files = await scanner.scan_folder(root, recursive=True)

        # Convert to relative paths for easier checking
        root_resolved = root.resolve()
        relative_paths = []
        for f in files:
            try:
                relative_paths.append(str(f.relative_to(root_resolved)))
            except ValueError:
                # Try with resolved path
                relative_paths.append(str(f.resolve().relative_to(root_resolved)))

        print(f"\nFiles found: {len(files)}")
        for path in sorted(relative_paths):
            print(f"  - {path}")

        # Assertions
        assert "src/main.py" in relative_paths, "Should find src/main.py"
        assert "src/utils.py" in relative_paths, "Should find src/utils.py"
        assert "docs/README.md" in relative_paths, "Should find docs/README.md"

        # Check exclusions
        assert not any(".venv" in p for p in relative_paths), "Should not find .venv files"
        assert not any(".uv-cache" in p for p in relative_paths), "Should not find .uv-cache files"
        assert not any(
            "node_modules" in p for p in relative_paths
        ), "Should not find node_modules files"
        assert not any(
            "__pycache__" in p for p in relative_paths
        ), "Should not find __pycache__ files"
        assert not any(".pyc" in p for p in relative_paths), "Should not find .pyc files"

        # Should find exactly 3 files
        assert len(files) == 3, f"Should find exactly 3 files, found {len(files)}"

        print("\n✅ All tests passed! Two-level exclusion is working correctly.")
        return True


if __name__ == "__main__":
    # Run the test
    import asyncio

    asyncio.run(test_exclusion())

    print("\n" + "=" * 60)
    print("Implementation ready to be integrated into indexer.py")
    print("=" * 60)
