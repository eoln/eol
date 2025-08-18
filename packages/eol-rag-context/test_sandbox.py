#!/usr/bin/env python
"""Test MCP server sandbox restrictions."""

import os
import sys
from pathlib import Path


def test_write_locations():
    """Test which locations are writable."""

    test_locations = [
        ("Current dir", Path.cwd() / "test.txt"),
        ("Relative path", Path("./test.txt")),
        ("Home dir", Path.home() / ".eol-test.txt"),
        ("Temp dir", Path("/tmp/eol-test.txt")),
        ("Data dir", Path("data") / "test.txt"),
        ("Absolute home", Path.home() / ".eol-rag-context" / "test.txt"),
    ]

    results = []
    for name, path in test_locations:
        try:
            # Try to create parent directory
            if path.parent != path.parent.parent:
                path.parent.mkdir(parents=True, exist_ok=True)

            # Try to write file
            path.write_text("test")
            path.unlink()  # Clean up
            results.append(f"✅ {name}: {path} - WRITABLE")
        except Exception as e:
            results.append(f"❌ {name}: {path} - {type(e).__name__}: {e}")

    return results


if __name__ == "__main__":
    print("Testing MCP Server Sandbox Restrictions")
    print("=" * 50)
    print(f"Current working directory: {os.getcwd()}")
    print(f"Process ID: {os.getpid()}")
    print(f"User: {os.getenv('USER', 'unknown')}")
    print("=" * 50)

    results = test_write_locations()
    for result in results:
        print(result)
