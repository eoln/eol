#!/usr/bin/env python3
"""Quick Start Example for EOL RAG Context.

This example demonstrates basic usage of the EOL RAG Context MCP server.

"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eol.rag_context import EOLRAGContextServer
from eol.rag_context.config import RAGConfig


async def main():
    """Run the quick start example."""

    print("=" * 60)
    print("EOL RAG Context - Quick Start Example")
    print("=" * 60)
    print()

    # Step 1: Initialize the server
    print("1. Initializing server...")
    config = RAGConfig()  # Use default configuration
    server = EOLRAGContextServer(config)

    try:
        await server.initialize()
        print("   ✅ Server initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
        print(
            "   Make sure Redis is running: docker run -d -p 6379:6379 redis/redis-stack:latest"
        )
        return

    print()

    # Step 2: Index a directory
    print("2. Indexing current directory...")
    current_dir = Path.cwd()

    # Find Python files to index
    py_files = list(current_dir.glob("*.py"))[:5]  # Limit to 5 files for demo

    if py_files:
        for file in py_files:
            result = await server.index_directory(str(file))
            print(f"   📄 Indexed {file.name}: {result.get('chunks', 0)} chunks")
    else:
        print("   ⚠️  No Python files found in current directory")

    print()

    # Step 3: Search for context
    print("3. Searching for context...")
    queries = [
        "initialize server",
        "index documents",
        "search context",
        "async function",
    ]

    for query in queries:
        print(f"\n   Query: '{query}'")
        results = await server.search_context(query, limit=3)

        if results:
            for i, result in enumerate(results, 1):
                print(f"   Result {i} (score: {result.get('score', 0):.2f}):")
                content = result.get("content", "")[:100]
                print(f"      {content}...")
        else:
            print("      No results found")

    print()

    # Step 4: Get statistics
    print("4. Server Statistics:")
    stats = await server.get_stats()

    print(f"   📊 Indexer:")
    print(f"      - Documents: {stats['indexer'].get('total_documents', 0)}")
    print(f"      - Chunks: {stats['indexer'].get('total_chunks', 0)}")

    print(f"   💾 Cache:")
    print(f"      - Queries: {stats['cache'].get('queries', 0)}")
    print(f"      - Hit Rate: {stats['cache'].get('hit_rate', 0):.1%}")

    print(f"   🔗 Knowledge Graph:")
    print(f"      - Nodes: {stats['graph'].get('nodes', 0)}")
    print(f"      - Edges: {stats['graph'].get('edges', 0)}")

    print()
    print("=" * 60)
    print("Quick Start Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Try indexing your own project directory")
    print("2. Experiment with different search queries")
    print("3. Enable file watching for real-time updates")
    print("4. Check out the advanced examples")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
