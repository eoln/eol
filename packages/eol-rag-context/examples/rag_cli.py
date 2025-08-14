#!/usr/bin/env python3
"""
RAG CLI - Command-line interface for EOL RAG Context

A simple CLI tool for indexing and searching with RAG context.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

from eol.rag_context import EOLRAGContextServer
from eol.rag_context.config import RAGConfig

app = typer.Typer(help="EOL RAG Context CLI")
console = Console()

# Global server instance
_server: Optional[EOLRAGContextServer] = None


async def get_server() -> EOLRAGContextServer:
    """Get or create server instance."""
    global _server

    if _server is None:
        _server = EOLRAGContextServer()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Initializing server...", total=None)

            try:
                await _server.initialize()
            except Exception as e:
                console.print(f"[red]❌ Failed to initialize: {e}[/red]")
                console.print("[yellow]Make sure Redis is running:[/yellow]")
                console.print("  docker run -d -p 6379:6379 redis/redis-stack:latest")
                raise typer.Exit(1)

    return _server


@app.command()
def index(
    path: Path = typer.Argument(..., help="Path to index"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Index recursively"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch for changes"),
    patterns: Optional[str] = typer.Option(
        None, "--patterns", "-p", help="File patterns (comma-separated)"
    ),
    ignore: Optional[str] = typer.Option(
        None, "--ignore", "-i", help="Ignore patterns (comma-separated)"
    ),
):
    """Index files or directories."""

    async def run_index():
        server = await get_server()

        # Parse patterns
        pattern_list = patterns.split(",") if patterns else None
        ignore_list = ignore.split(",") if ignore else None

        # Show what we're indexing
        console.print(f"\n[blue]📁 Indexing: {path}[/blue]")
        if pattern_list:
            console.print(f"   Patterns: {', '.join(pattern_list)}")
        if ignore_list:
            console.print(f"   Ignoring: {', '.join(ignore_list)}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Indexing...", total=None)

            result = await server.index_directory(
                str(path), recursive=recursive, patterns=pattern_list, ignore=ignore_list
            )

            progress.update(task, completed=True)

        # Show results
        console.print(f"\n[green]✅ Indexing complete![/green]")
        console.print(f"   Files: {result.get('indexed_files', 0)}")
        console.print(f"   Chunks: {result.get('total_chunks', 0)}")
        console.print(f"   Source ID: {result.get('source_id', 'N/A')}")

        if result.get("errors"):
            console.print(f"\n[yellow]⚠️  Errors: {len(result['errors'])}[/yellow]")
            for error in result["errors"][:5]:
                console.print(f"   - {error}")

        # Start watching if requested
        if watch:
            console.print(f"\n[blue]👁️  Watching for changes...[/blue]")
            watch_result = await server.watch_directory(
                str(path), patterns=pattern_list, ignore=ignore_list
            )
            console.print(f"   Watch ID: {watch_result['watch_id']}")
            console.print("[dim]Press Ctrl+C to stop watching[/dim]")

            try:
                await asyncio.sleep(float("inf"))
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopping watch...[/yellow]")

    asyncio.run(run_index())


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of results"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    show_metadata: bool = typer.Option(False, "--metadata", "-m", help="Show metadata"),
    file_type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by file type"),
):
    """Search for context."""

    async def run_search():
        server = await get_server()

        # Build filters
        filters = {}
        if file_type:
            filters["file_type"] = file_type

        # Perform search
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"Searching for: {query}", total=None)

            results = await server.search_context(
                query, limit=limit, filters=filters if filters else None
            )

        # Output results
        if json_output:
            console.print_json(json.dumps(results, indent=2))
        else:
            if not results:
                console.print("[yellow]No results found[/yellow]")
                return

            console.print(f"\n[green]Found {len(results)} results:[/green]\n")

            for i, result in enumerate(results, 1):
                # Create result panel
                score = result.get("score", 0)
                content = result.get("content", "")[:300]
                metadata = result.get("metadata", {})

                # Format content with syntax highlighting if it's code
                file_type = metadata.get("file_type", "text")
                if file_type == "code":
                    lang = metadata.get("language", "python")
                    content_display = Syntax(content, lang, theme="monokai", line_numbers=True)
                else:
                    content_display = content + "..."

                # Create panel
                title = f"Result {i} - Score: {score:.2%}"
                if metadata.get("source"):
                    source = Path(metadata["source"]).name
                    title += f" - {source}"

                panel = Panel(
                    content_display, title=title, border_style="blue" if i == 1 else "dim"
                )
                console.print(panel)

                # Show metadata if requested
                if show_metadata:
                    meta_table = Table(show_header=False, box=None, padding=(0, 2))
                    meta_table.add_column("Key", style="cyan")
                    meta_table.add_column("Value")

                    for key, value in metadata.items():
                        if key != "content":
                            meta_table.add_row(key, str(value))

                    console.print(meta_table)
                    console.print()

    asyncio.run(run_search())


@app.command()
def stats():
    """Show server statistics."""

    async def run_stats():
        server = await get_server()
        stats = await server.get_stats()

        # Create stats table
        table = Table(title="EOL RAG Context Statistics", show_header=True)
        table.add_column("Component", style="cyan")
        table.add_column("Metric", style="magenta")
        table.add_column("Value", style="green")

        # Indexer stats
        indexer = stats.get("indexer", {})
        table.add_row("Indexer", "Documents", str(indexer.get("total_documents", 0)))
        table.add_row("", "Chunks", str(indexer.get("total_chunks", 0)))
        table.add_row("", "Sources", str(len(indexer.get("sources", []))))

        # Cache stats
        cache = stats.get("cache", {})
        table.add_row("Cache", "Queries", str(cache.get("queries", 0)))
        table.add_row("", "Hits", str(cache.get("hits", 0)))
        table.add_row("", "Hit Rate", f"{cache.get('hit_rate', 0):.1%}")

        # Knowledge Graph stats
        graph = stats.get("graph", {})
        table.add_row("Graph", "Nodes", str(graph.get("nodes", 0)))
        table.add_row("", "Edges", str(graph.get("edges", 0)))

        console.print(table)

    asyncio.run(run_stats())


@app.command()
def clear(
    cache_only: bool = typer.Option(False, "--cache-only", help="Clear cache only"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Clear indexed data and/or cache."""

    async def run_clear():
        if not confirm:
            if cache_only:
                confirm_msg = "Clear cache?"
            else:
                confirm_msg = "Clear all indexed data and cache?"

            if not typer.confirm(confirm_msg):
                console.print("[yellow]Cancelled[/yellow]")
                return

        server = await get_server()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if cache_only:
                progress.add_task("Clearing cache...", total=None)
                result = await server.clear_cache()
            else:
                progress.add_task("Clearing all data...", total=None)
                # Clear all sources
                stats = await server.get_stats()
                sources = stats.get("indexer", {}).get("sources", [])

                for source in sources:
                    await server.remove_source(source.get("source_id"))

                result = await server.clear_cache()

        console.print("[green]✅ Clear complete![/green]")
        if result:
            console.print(f"   Status: {result.get('status', 'unknown')}")

    asyncio.run(run_clear())


@app.command()
def watch(
    path: Path = typer.Argument(..., help="Path to watch"),
    patterns: Optional[str] = typer.Option(
        None, "--patterns", "-p", help="File patterns (comma-separated)"
    ),
    ignore: Optional[str] = typer.Option(
        None, "--ignore", "-i", help="Ignore patterns (comma-separated)"
    ),
):
    """Watch a directory for changes."""

    async def run_watch():
        server = await get_server()

        # Parse patterns
        pattern_list = patterns.split(",") if patterns else None
        ignore_list = ignore.split(",") if ignore else None

        console.print(f"\n[blue]👁️  Starting file watcher for: {path}[/blue]")

        result = await server.watch_directory(str(path), patterns=pattern_list, ignore=ignore_list)

        console.print(f"[green]✅ Watching started![/green]")
        console.print(f"   Watch ID: {result['watch_id']}")
        console.print(f"   Path: {result['path']}")
        console.print("[dim]Press Ctrl+C to stop watching[/dim]\n")

        try:
            # Keep running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping watch...[/yellow]")
            await server.unwatch_directory(result["watch_id"])
            console.print("[green]✅ Watch stopped[/green]")

    asyncio.run(run_watch())


@app.command()
def serve(
    host: str = typer.Option("localhost", "--host", help="Server host"),
    port: int = typer.Option(8080, "--port", help="Server port"),
):
    """Start MCP server (for Claude Desktop integration)."""

    async def run_serve():
        server = await get_server()

        console.print(f"\n[blue]🚀 Starting MCP server on {host}:{port}[/blue]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")

        try:
            await server.run()
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down...[/yellow]")

    asyncio.run(run_serve())


@app.callback()
def main():
    """
    EOL RAG Context CLI - Intelligent context management for AI applications.

    Examples:
        # Index a directory
        rag_cli index /path/to/project

        # Search for context
        rag_cli search "authentication flow"

        # Watch for changes
        rag_cli watch /path/to/project --patterns "*.py,*.md"

        # Show statistics
        rag_cli stats
    """
    pass


if __name__ == "__main__":
    app()
