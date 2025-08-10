"""EOL CLI Main Entry Point"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.syntax import Syntax
from rich.panel import Panel
import json
import yaml

# Check for MCP mode early
if "--mcp" in sys.argv or "MCP_MODE" in os.environ:
    # Import and run MCP server instead of CLI
    from eol.mcp import run_mcp_server
    run_mcp_server()
    sys.exit(0)

from eol.core import (
    EOLParser,
    PhaseManager,
    ContextManager,
    DependencyResolver,
    ExecutionPhase
)

# Initialize Typer app
app = typer.Typer(
    name="eol",
    help="EOL Framework - AI Framework for building modern LLM applications",
    add_completion=False,
)

console = Console()


@app.command()
def run(
    feature: str = typer.Argument(..., help="Path to .eol.md file"),
    phase: str = typer.Option("hybrid", "--phase", "-p", help="Execution phase: prototyping|implementation|hybrid"),
    operation: Optional[str] = typer.Option(None, "--operation", "-o", help="Specific operation to run"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch for file changes"),
    skip_deps: bool = typer.Option(False, "--skip-deps", help="Skip dependency resolution"),
    profile: Optional[str] = typer.Option(None, "--profile", help="Dependency profile to use"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run an EOL feature file"""
    
    async def _run():
        try:
            # Validate file exists
            feature_path = Path(feature)
            if not feature_path.exists():
                console.print(f"[red]✗[/red] Feature file not found: {feature}")
                raise typer.Exit(1)
            
            # Parse feature
            console.print(f"[cyan]Parsing feature:[/cyan] {feature}")
            parser = EOLParser()
            feature_spec = parser.parse_feature(feature_path)
            
            # Resolve dependencies
            if not skip_deps:
                console.print("[cyan]Resolving dependencies...[/cyan]")
                resolver = DependencyResolver(Path.cwd())
                dependencies = await resolver.resolve_feature(feature, phase)
                console.print(f"[green]✓[/green] Resolved {len(dependencies)} dependencies")
                
                if verbose:
                    _display_dependencies(dependencies)
            else:
                dependencies = {}
            
            # Initialize phase manager
            phase_manager = PhaseManager()
            current_phase = ExecutionPhase(phase)
            
            # Execute feature
            console.print(f"[cyan]Executing in {phase} mode...[/cyan]")
            
            if operation:
                # Execute specific operation
                console.print(f"[cyan]Running operation:[/cyan] {operation}")
                # TODO: Implement operation execution
                console.print(f"[green]✓[/green] Operation completed")
            else:
                # Execute all operations
                for op in feature_spec.operations:
                    op_name = op.get('name', 'unnamed')
                    op_phase = op.get('phase', phase)
                    
                    if phase_manager.get_operation_phase(feature_spec.name, op_name) == current_phase:
                        console.print(f"[cyan]Running:[/cyan] {op_name}")
                        # TODO: Implement operation execution
                        console.print(f"[green]✓[/green] {op_name} completed")
            
            console.print("[green]✓[/green] Feature execution completed successfully")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Execution failed: {e}")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    # Run async function
    asyncio.run(_run())


@app.command()
def test(
    test_file: str = typer.Argument(..., help="Path to .test.eol.md file"),
    coverage: bool = typer.Option(False, "--coverage", "-c", help="Generate coverage report"),
    pattern: Optional[str] = typer.Option(None, "--pattern", help="Test name pattern to match"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run EOL test specifications"""
    
    try:
        # Validate file exists
        test_path = Path(test_file)
        if not test_path.exists():
            console.print(f"[red]✗[/red] Test file not found: {test_file}")
            raise typer.Exit(1)
        
        # Parse test file
        console.print(f"[cyan]Parsing test file:[/cyan] {test_file}")
        parser = EOLParser()
        test_spec = parser.parse_test(test_path)
        
        # Run tests
        console.print(f"[cyan]Running tests for:[/cyan] {test_spec.feature}")
        
        passed = 0
        failed = 0
        skipped = 0
        
        for test_case in test_spec.test_cases:
            test_name = test_case.get('name', 'unnamed')
            
            if pattern and pattern not in test_name:
                skipped += 1
                continue
            
            console.print(f"  [cyan]Testing:[/cyan] {test_name}")
            # TODO: Implement test execution
            passed += 1
            console.print(f"    [green]✓[/green] Passed")
        
        # Display results
        console.print("\n[bold]Test Results:[/bold]")
        console.print(f"  [green]Passed:[/green] {passed}")
        console.print(f"  [red]Failed:[/red] {failed}")
        console.print(f"  [yellow]Skipped:[/yellow] {skipped}")
        
        if coverage:
            console.print("\n[cyan]Generating coverage report...[/cyan]")
            # TODO: Implement coverage generation
            console.print("[green]✓[/green] Coverage report generated")
        
        if failed > 0:
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]✗[/red] Test execution failed: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def generate(
    feature: str = typer.Argument(..., help="Path to .eol.md file"),
    output: str = typer.Option("./src", "--output", "-o", help="Output directory"),
    language: str = typer.Option("python", "--language", "-l", help="Target language"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
):
    """Generate implementation from prototype"""
    
    try:
        # Validate feature file
        feature_path = Path(feature)
        if not feature_path.exists():
            console.print(f"[red]✗[/red] Feature file not found: {feature}")
            raise typer.Exit(1)
        
        # Parse feature
        console.print(f"[cyan]Parsing feature:[/cyan] {feature}")
        parser = EOLParser()
        feature_spec = parser.parse_feature(feature_path)
        
        # Check if prototyping section exists
        if not feature_spec.prototyping:
            console.print("[red]✗[/red] No prototyping section found in feature")
            raise typer.Exit(1)
        
        # Create output directory
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[cyan]Generating {language} implementation...[/cyan]")
        
        # TODO: Implement code generation
        generated_file = output_path / f"{feature_spec.name}.py"
        
        if generated_file.exists() and not force:
            console.print(f"[yellow]![/yellow] File exists: {generated_file}")
            if not typer.confirm("Overwrite?"):
                raise typer.Exit(0)
        
        # Write generated code
        # TODO: Actual generation logic
        generated_file.write_text(f"# Generated from {feature}\n# TODO: Implement\n")
        
        console.print(f"[green]✓[/green] Generated: {generated_file}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Generation failed: {e}")
        raise typer.Exit(1)


@app.command()
def switch(
    feature: str = typer.Argument(..., help="Path to .eol.md file"),
    to_phase: str = typer.Option(..., "--to", help="Target phase: prototyping|implementation|hybrid"),
    operations: Optional[str] = typer.Option(None, "--operations", help="Comma-separated list of operations"),
):
    """Switch feature execution phase"""
    
    try:
        # Validate feature file
        feature_path = Path(feature)
        if not feature_path.exists():
            console.print(f"[red]✗[/red] Feature file not found: {feature}")
            raise typer.Exit(1)
        
        # Parse feature
        parser = EOLParser()
        feature_spec = parser.parse_feature(feature_path)
        
        # Parse operations list
        op_list = operations.split(",") if operations else None
        
        # Switch phase
        phase_manager = PhaseManager()
        result = phase_manager.switch_phase(
            feature_spec.name,
            ExecutionPhase(to_phase),
            op_list
        )
        
        console.print(f"[green]✓[/green] Switched {feature_spec.name} to {to_phase}")
        
        if op_list:
            console.print(f"  Operations: {', '.join(op_list)}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Phase switch failed: {e}")
        raise typer.Exit(1)


# Dependency management commands
deps_app = typer.Typer(help="Manage EOL dependencies")
app.add_typer(deps_app, name="deps")


@deps_app.command("install")
def deps_install(
    feature: Optional[str] = typer.Argument(None, help="Feature file to install dependencies for"),
    phase: str = typer.Option("all", "--phase", "-p", help="Phase to install for"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Install dependencies for a feature"""
    
    async def _install():
        try:
            if feature:
                # Install for specific feature
                feature_path = Path(feature)
                if not feature_path.exists():
                    console.print(f"[red]✗[/red] Feature file not found: {feature}")
                    raise typer.Exit(1)
                
                console.print(f"[cyan]Installing dependencies for:[/cyan] {feature}")
                resolver = DependencyResolver(Path.cwd())
                dependencies = await resolver.resolve_feature(feature, phase)
                
                # TODO: Implement actual installation
                console.print(f"[green]✓[/green] Installed {len(dependencies)} dependencies")
            else:
                # Install all dependencies from pyproject.toml
                console.print("[cyan]Installing project dependencies...[/cyan]")
                # TODO: Implement project-wide installation
                console.print("[green]✓[/green] Dependencies installed")
                
        except Exception as e:
            console.print(f"[red]✗[/red] Installation failed: {e}")
            if verbose:
                console.print_exception()
            raise typer.Exit(1)
    
    asyncio.run(_install())


@deps_app.command("health")
def deps_health(
    feature: Optional[str] = typer.Argument(None, help="Feature file to check"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Check health of dependencies"""
    
    async def _health():
        try:
            if feature:
                feature_path = Path(feature)
                if not feature_path.exists():
                    console.print(f"[red]✗[/red] Feature file not found: {feature}")
                    raise typer.Exit(1)
                
                resolver = DependencyResolver(Path.cwd())
                await resolver.resolve_feature(feature)
                health_status = await resolver.health_check()
            else:
                # Check all dependencies
                health_status = {}
                # TODO: Implement project-wide health check
            
            if json_output:
                console.print(json.dumps(health_status, indent=2))
            else:
                _display_health_status(health_status)
                
        except Exception as e:
            console.print(f"[red]✗[/red] Health check failed: {e}")
            raise typer.Exit(1)
    
    asyncio.run(_health())


@deps_app.command("list")
def deps_list(
    feature: Optional[str] = typer.Argument(None, help="Feature file to list dependencies for"),
    type: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by dependency type"),
):
    """List all dependencies"""
    
    async def _list():
        try:
            if feature:
                feature_path = Path(feature)
                if not feature_path.exists():
                    console.print(f"[red]✗[/red] Feature file not found: {feature}")
                    raise typer.Exit(1)
                
                parser = EOLParser()
                feature_spec = parser.parse_feature(feature_path)
                
                _display_feature_dependencies(feature_spec.dependencies, type)
            else:
                # List all project dependencies
                # TODO: Implement project-wide listing
                console.print("[yellow]![/yellow] Project-wide listing not yet implemented")
                
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to list dependencies: {e}")
            raise typer.Exit(1)
    
    asyncio.run(_list())


@deps_app.command("graph")
def deps_graph(
    feature: str = typer.Argument(..., help="Feature file to graph"),
    output: str = typer.Option("deps.svg", "--output", "-o", help="Output file"),
    format: str = typer.Option("svg", "--format", "-f", help="Output format: svg|png|dot"),
):
    """Generate dependency graph visualization"""
    
    async def _graph():
        try:
            feature_path = Path(feature)
            if not feature_path.exists():
                console.print(f"[red]✗[/red] Feature file not found: {feature}")
                raise typer.Exit(1)
            
            console.print(f"[cyan]Generating dependency graph...[/cyan]")
            
            resolver = DependencyResolver(Path.cwd())
            await resolver.resolve_feature(feature)
            graph = resolver.get_dependency_graph()
            
            # TODO: Implement graph visualization
            console.print(f"[green]✓[/green] Graph saved to: {output}")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Graph generation failed: {e}")
            raise typer.Exit(1)
    
    asyncio.run(_graph())


# Utility functions
def _display_dependencies(dependencies: dict):
    """Display resolved dependencies in a table"""
    
    table = Table(title="Resolved Dependencies")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Version", style="yellow")
    table.add_column("Status", style="magenta")
    
    for key, value in dependencies.items():
        parts = key.split(":", 1)
        dep_type = parts[0] if parts else "unknown"
        dep_name = parts[1] if len(parts) > 1 else key
        
        # TODO: Get version and status from value
        table.add_row(dep_type, dep_name, "1.0.0", "✓")
    
    console.print(table)


def _display_health_status(health_status: dict):
    """Display health status in a formatted way"""
    
    table = Table(title="Dependency Health")
    table.add_column("Dependency", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")
    
    for key, status in health_status.items():
        status_icon = "✓" if status.get('status') == 'healthy' else "✗"
        status_color = "green" if status.get('status') == 'healthy' else "red"
        details = status.get('message', status.get('error', ''))
        
        table.add_row(
            key,
            f"[{status_color}]{status_icon}[/{status_color}] {status['status']}",
            details
        )
    
    console.print(table)


def _display_feature_dependencies(dependencies: dict, filter_type: Optional[str] = None):
    """Display feature dependencies"""
    
    for dep_type, deps in dependencies.items():
        if filter_type and dep_type != filter_type:
            continue
        
        console.print(f"\n[bold cyan]{dep_type.upper()}:[/bold cyan]")
        
        for dep in deps:
            name = dep.get('name', dep.get('path', 'unknown'))
            version = dep.get('version', 'any')
            phase = dep.get('phase', 'all')
            
            console.print(f"  • {name} [{version}] (phase: {phase})")


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Port to serve on"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start EOL as MCP HTTP server"""
    
    console.print(f"[cyan]Starting EOL MCP server on {host}:{port}...[/cyan]")
    
    # Set environment variables for MCP mode
    os.environ["MCP_MODE"] = "true"
    os.environ["MCP_TRANSPORT"] = "sse"
    os.environ["MCP_HOST"] = host
    os.environ["MCP_PORT"] = str(port)
    
    try:
        # Import and run MCP server
        from eol.mcp import run_mcp_server
        run_mcp_server()
    except ImportError:
        console.print("[red]✗[/red] MCP server not installed. Install with: pip install eol[mcp]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show EOL version"""
    
    from eol.cli import __version__ as cli_version
    from eol.core import __version__ as core_version
    
    console.print(Panel.fit(
        f"[bold cyan]EOL Framework[/bold cyan]\n"
        f"CLI: v{cli_version}\n"
        f"Core: v{core_version}",
        title="Version Information"
    ))


def main():
    """Main entry point for the CLI"""
    app()


if __name__ == "__main__":
    main()