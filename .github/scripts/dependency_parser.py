#!/usr/bin/env python3
"""Parse dependencies from pyproject.toml for security scanning."""

import sys
from pathlib import Path

try:
    import tomli
except ImportError:
    import tomllib as tomli  # Python 3.11+


def extract_dependencies(pyproject_file: str, output_file: str = None) -> int:
    """Extract dependencies from pyproject.toml.

    Args:
        pyproject_file: Path to pyproject.toml
        output_file: Optional output file for requirements

    Returns:
        0 on success, 1 on failure
    """
    try:
        if not Path(pyproject_file).exists():
            print(f"❌ pyproject.toml not found: {pyproject_file}")
            return 1

        with open(pyproject_file, "rb") as f:
            data = tomli.load(f)

        # Extract dependencies
        project = data.get("project", {})
        dependencies = project.get("dependencies", [])

        # Extract optional dependencies
        optional_deps = project.get("optional-dependencies", {})
        all_deps = list(dependencies)

        for group, deps in optional_deps.items():
            print(f"📦 Found {len(deps)} dependencies in group '{group}'")
            all_deps.extend(deps)

        if not all_deps:
            print("⚠️ No dependencies found in pyproject.toml")
            return 0

        print(f"📊 Total dependencies: {len(all_deps)}")

        # Write to output file if specified
        if output_file:
            with open(output_file, "w") as f:
                for dep in all_deps:
                    f.write(dep + "\n")
            print(f"✅ Dependencies written to {output_file}")
        else:
            # Print dependencies to stdout
            for dep in all_deps:
                print(dep)

        return 0

    except Exception as e:
        print(f"❌ Error parsing pyproject.toml: {e}")
        return 1


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python dependency_parser.py <pyproject.toml> [output_file]")
        sys.exit(1)

    pyproject_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    exit_code = extract_dependencies(pyproject_file, output_file)
    sys.exit(exit_code)
