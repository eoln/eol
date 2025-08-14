#!/usr/bin/env python3
"""
Code Assistant Example

This example shows how to build an AI code assistant using EOL RAG Context.
It indexes a codebase and provides intelligent context for code-related queries.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eol.rag_context import EOLRAGContextServer
from eol.rag_context.config import RAGConfig


class CodeAssistant:
    """AI-powered code assistant with RAG context."""

    def __init__(self, project_path: Path):
        """Initialize the code assistant.

        Args:
            project_path: Path to the project to analyze
        """
        self.project_path = project_path
        self.server = None
        self.indexed_files = 0
        self.total_chunks = 0

    async def initialize(self):
        """Initialize the RAG server and index the project."""
        print("🤖 Code Assistant Initializing...")

        # Configure for code analysis
        config = RAGConfig()
        config.document.supported_extensions.extend(
            [
                ".py",
                ".js",
                ".ts",
                ".jsx",
                ".tsx",
                ".java",
                ".cpp",
                ".c",
                ".h",
                ".go",
                ".rs",
                ".rb",
                ".php",
                ".swift",
            ]
        )
        config.chunking.max_chunk_size = 500  # Smaller chunks for code
        config.chunking.chunk_overlap = 100

        self.server = EOLRAGContextServer(config)

        try:
            await self.server.initialize()
            print("✅ Server initialized")
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            raise

        # Index the project
        await self.index_project()

    async def index_project(self):
        """Index the entire project."""
        print(f"\n📁 Indexing project: {self.project_path}")

        result = await self.server.index_directory(
            str(self.project_path),
            recursive=True,
            patterns=["*.py", "*.js", "*.md", "*.yaml", "*.json"],
            ignore=["__pycache__", ".git", "node_modules", ".venv", "venv"],
        )

        self.indexed_files = result.get("indexed_files", 0)
        self.total_chunks = result.get("total_chunks", 0)

        print(f"✅ Indexed {self.indexed_files} files")
        print(f"📦 Created {self.total_chunks} searchable chunks")

    async def answer_question(self, question: str) -> str:
        """Answer a code-related question using RAG context.

        Args:
            question: The user's question

        Returns:
            An answer based on the codebase context
        """
        # Search for relevant context
        contexts = await self.server.search_context(
            question, limit=5, filters={"file_type": ["code", "markdown"]}
        )

        if not contexts:
            return "I couldn't find relevant information in the codebase for your question."

        # Build response with context
        response = f"Based on the codebase analysis, here's what I found:\n\n"

        for i, ctx in enumerate(contexts, 1):
            source = Path(ctx["metadata"].get("source", "unknown")).name
            score = ctx.get("score", 0)

            response += f"**{i}. From {source} (relevance: {score:.0%}):**\n"
            response += f"```\n{ctx['content'][:300]}...\n```\n\n"

        return response

    async def find_implementations(self, entity: str) -> List[Dict[str, Any]]:
        """Find implementations of a class, function, or interface.

        Args:
            entity: Name of the entity to find

        Returns:
            List of implementations found
        """
        # Use knowledge graph to find entity relationships
        graph = await self.server.query_knowledge_graph(entity, max_depth=2)

        implementations = []

        for node in graph.get("entities", []):
            if entity.lower() in node.get("name", "").lower():
                implementations.append(
                    {
                        "name": node["name"],
                        "type": node.get("type", "unknown"),
                        "file": node.get("metadata", {}).get("source", "unknown"),
                        "line": node.get("metadata", {}).get("line", 0),
                    }
                )

        return implementations

    async def suggest_improvements(self, code: str) -> List[str]:
        """Suggest improvements for a code snippet.

        Args:
            code: The code to analyze

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        # Search for similar patterns
        similar = await self.server.search_context(code, limit=3, filters={"file_type": "code"})

        for ctx in similar:
            # Look for better implementations
            if ctx.get("score", 0) > 0.8:
                suggestions.append(
                    f"Consider this pattern from {ctx['metadata'].get('source', 'codebase')}:\n"
                    f"{ctx['content'][:200]}"
                )

        # Search for best practices
        practices = await self.server.search_context(
            f"best practices {code[:50]}", limit=2, filters={"file_type": "markdown"}
        )

        for practice in practices:
            suggestions.append(f"Best practice: {practice['content'][:150]}")

        return suggestions

    async def interactive_session(self):
        """Run an interactive Q&A session."""
        print("\n" + "=" * 60)
        print("🤖 Code Assistant Ready!")
        print("=" * 60)
        print(f"Project: {self.project_path}")
        print(f"Files indexed: {self.indexed_files}")
        print(f"Searchable chunks: {self.total_chunks}")
        print("\nType 'help' for commands or 'quit' to exit")
        print("-" * 60)

        commands = {
            "help": self.show_help,
            "stats": self.show_stats,
            "reindex": self.index_project,
            "quit": lambda: None,
            "exit": lambda: None,
        }

        while True:
            try:
                user_input = input("\n💬 You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit"]:
                    print("👋 Goodbye!")
                    break

                if user_input.lower() in commands:
                    result = commands[user_input.lower()]()
                    if asyncio.iscoroutine(result):
                        await result
                    continue

                # Check for special commands
                if user_input.startswith("find "):
                    entity = user_input[5:]
                    print(f"\n🔍 Searching for '{entity}'...")
                    implementations = await self.find_implementations(entity)

                    if implementations:
                        print(f"Found {len(implementations)} implementations:")
                        for impl in implementations:
                            print(
                                f"  📄 {impl['name']} ({impl['type']}) - {impl['file']}:{impl['line']}"
                            )
                    else:
                        print("No implementations found")
                    continue

                if user_input.startswith("improve "):
                    code = user_input[8:]
                    print("\n💡 Analyzing code for improvements...")
                    suggestions = await self.suggest_improvements(code)

                    if suggestions:
                        print("Suggestions:")
                        for i, suggestion in enumerate(suggestions, 1):
                            print(f"\n{i}. {suggestion}")
                    else:
                        print("No specific suggestions found")
                    continue

                # Regular question
                print("\n🤔 Thinking...")
                answer = await self.answer_question(user_input)
                print(f"\n🤖 Assistant: {answer}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    def show_help(self):
        """Show help information."""
        print("\n📚 Available Commands:")
        print("  help          - Show this help message")
        print("  stats         - Show indexing statistics")
        print("  reindex       - Re-index the project")
        print("  find <name>   - Find implementations of a class/function")
        print("  improve <code> - Get improvement suggestions for code")
        print("  quit/exit     - Exit the assistant")
        print("\nOr just ask any question about the codebase!")

    async def show_stats(self):
        """Show current statistics."""
        stats = await self.server.get_stats()

        print("\n📊 Statistics:")
        print(f"  Documents: {stats['indexer'].get('total_documents', 0)}")
        print(f"  Chunks: {stats['indexer'].get('total_chunks', 0)}")
        print(f"  Cache Hit Rate: {stats['cache'].get('hit_rate', 0):.1%}")
        print(f"  Graph Nodes: {stats['graph'].get('nodes', 0)}")
        print(f"  Graph Edges: {stats['graph'].get('edges', 0)}")


async def main():
    """Run the code assistant example."""
    import argparse

    parser = argparse.ArgumentParser(description="AI Code Assistant")
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to the project to analyze (default: current directory)",
    )
    args = parser.parse_args()

    project_path = Path(args.path).resolve()

    if not project_path.exists():
        print(f"❌ Error: Path '{project_path}' does not exist")
        return

    # Create and run the assistant
    assistant = CodeAssistant(project_path)

    try:
        await assistant.initialize()
        await assistant.interactive_session()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("\nMake sure Redis is running:")
        print("  docker run -d -p 6379:6379 redis/redis-stack:latest")


if __name__ == "__main__":
    asyncio.run(main())
