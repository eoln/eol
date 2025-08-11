#!/usr/bin/env python3
"""
Documentation validation script.

This script validates the documentation quality and coverage for the EOL RAG Context project.
It checks docstring coverage, validates links, and ensures documentation standards are met.
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import importlib.util
import inspect
import argparse
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from rich.progress import track

console = Console()

class DocstringAnalyzer(ast.NodeVisitor):
    """Analyzes Python files for docstring coverage."""
    
    def __init__(self):
        self.stats = {
            'modules': {'total': 0, 'documented': 0},
            'classes': {'total': 0, 'documented': 0},
            'methods': {'total': 0, 'documented': 0},
            'functions': {'total': 0, 'documented': 0}
        }
        self.missing_docs = []
        self.current_file = None
        
    def visit_Module(self, node):
        """Visit module node."""
        self.stats['modules']['total'] += 1
        if ast.get_docstring(node):
            self.stats['modules']['documented'] += 1
        else:
            self.missing_docs.append(f"{self.current_file}: Module docstring missing")
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        """Visit class definition."""
        if not node.name.startswith('_'):  # Skip private classes
            self.stats['classes']['total'] += 1
            if ast.get_docstring(node):
                self.stats['classes']['documented'] += 1
            else:
                self.missing_docs.append(f"{self.current_file}:{node.lineno}: Class '{node.name}' missing docstring")
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        """Visit function definition."""
        if not node.name.startswith('_'):  # Skip private methods
            category = 'methods' if self._is_method(node) else 'functions'
            self.stats[category]['total'] += 1
            if ast.get_docstring(node):
                self.stats[category]['documented'] += 1
            else:
                self.missing_docs.append(f"{self.current_file}:{node.lineno}: Function '{node.name}' missing docstring")
        self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node):
        """Visit async function definition."""
        self.visit_FunctionDef(node)  # Treat async functions the same way
        
    def _is_method(self, node):
        """Check if a function is a method (inside a class)."""
        for parent in ast.walk(node):
            if isinstance(parent, ast.ClassDef):
                return True
        return False


def analyze_file(file_path: Path) -> DocstringAnalyzer:
    """Analyze a single Python file for docstring coverage."""
    analyzer = DocstringAnalyzer()
    try:
        analyzer.current_file = str(file_path.relative_to(Path.cwd()))
    except ValueError:
        analyzer.current_file = str(file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        analyzer.visit(tree)
    except Exception as e:
        console.print(f"[red]Error analyzing {file_path}: {e}[/red]")
    
    return analyzer


def analyze_directory(directory: Path) -> Dict:
    """Analyze all Python files in a directory for docstring coverage."""
    total_stats = {
        'modules': {'total': 0, 'documented': 0},
        'classes': {'total': 0, 'documented': 0},
        'methods': {'total': 0, 'documented': 0},
        'functions': {'total': 0, 'documented': 0}
    }
    all_missing = []
    
    python_files = list(directory.rglob("*.py"))
    python_files = [f for f in python_files if not any(part.startswith('.') for part in f.parts)]
    python_files = [f for f in python_files if 'test' not in f.name.lower()]
    
    for file_path in track(python_files, description="Analyzing files..."):
        analyzer = analyze_file(file_path)
        
        # Aggregate stats
        for category in total_stats:
            total_stats[category]['total'] += analyzer.stats[category]['total']
            total_stats[category]['documented'] += analyzer.stats[category]['documented']
        
        all_missing.extend(analyzer.missing_docs)
    
    return total_stats, all_missing


def calculate_coverage(stats: Dict) -> float:
    """Calculate overall docstring coverage percentage."""
    total = sum(s['total'] for s in stats.values())
    documented = sum(s['documented'] for s in stats.values())
    
    if total == 0:
        return 100.0
    
    return (documented / total) * 100


def print_coverage_report(stats: Dict, missing: List[str], verbose: bool = False):
    """Print a formatted coverage report."""
    console.print("\n[bold blue]📊 Docstring Coverage Report[/bold blue]\n")
    
    # Create coverage table
    table = Table(title="Coverage by Category")
    table.add_column("Category", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Documented", justify="right", style="green")
    table.add_column("Missing", justify="right", style="red")
    table.add_column("Coverage", justify="right")
    
    for category, data in stats.items():
        total = data['total']
        documented = data['documented']
        missing_count = total - documented
        coverage = (documented / total * 100) if total > 0 else 100
        
        coverage_style = "green" if coverage >= 80 else "yellow" if coverage >= 60 else "red"
        table.add_row(
            category.capitalize(),
            str(total),
            str(documented),
            str(missing_count),
            f"[{coverage_style}]{coverage:.1f}%[/{coverage_style}]"
        )
    
    console.print(table)
    
    # Overall coverage
    overall = calculate_coverage(stats)
    overall_style = "green" if overall >= 80 else "yellow" if overall >= 60 else "red"
    console.print(f"\n[bold]Overall Coverage: [{overall_style}]{overall:.1f}%[/{overall_style}][/bold]")
    
    # Success criteria
    if overall >= 95:
        console.print("[green]✅ Exceeds 95% coverage target![/green]")
    elif overall >= 80:
        console.print("[yellow]⚠️  Good coverage, but below 95% target[/yellow]")
    else:
        console.print("[red]❌ Coverage below acceptable threshold[/red]")
    
    # Show missing docstrings if verbose
    if verbose and missing:
        console.print("\n[bold red]Missing Docstrings:[/bold red]")
        for item in missing[:20]:  # Show first 20
            console.print(f"  • {item}")
        if len(missing) > 20:
            console.print(f"  ... and {len(missing) - 20} more")


def validate_mkdocs_config() -> bool:
    """Validate MkDocs configuration file."""
    config_path = Path("mkdocs.yml")
    if not config_path.exists():
        console.print("[red]❌ mkdocs.yml not found[/red]")
        return False
    
    try:
        import yaml
        # Create a custom loader that ignores special tags
        class SafeLoaderIgnoreTags(yaml.SafeLoader):
            pass
        # Ignore all unknown tags
        SafeLoaderIgnoreTags.add_constructor(None, lambda loader, node: '')
        
        with open(config_path) as f:
            config = yaml.load(f, Loader=SafeLoaderIgnoreTags)
        
        # Check required sections
        required = ['site_name', 'theme', 'plugins', 'nav']
        missing = [r for r in required if r not in config]
        
        if missing:
            console.print(f"[red]❌ Missing required config sections: {missing}[/red]")
            return False
        
        # Check for mkdocstrings plugin
        plugins = config.get('plugins', [])
        has_mkdocstrings = any(
            p == 'mkdocstrings' or (isinstance(p, dict) and 'mkdocstrings' in p)
            for p in plugins
        )
        
        if not has_mkdocstrings:
            console.print("[yellow]⚠️  mkdocstrings plugin not configured[/yellow]")
            return False
        
        console.print("[green]✅ MkDocs configuration valid[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Error validating config: {e}[/red]")
        return False


def main():
    """Main entry point for documentation validation."""
    parser = argparse.ArgumentParser(description="Validate documentation coverage and quality")
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    parser.add_argument('--path', '-p', type=Path, default=Path('src/eol/rag_context'),
                       help='Path to analyze (default: src/eol/rag_context)')
    parser.add_argument('--min-coverage', '-m', type=float, default=80.0,
                       help='Minimum coverage percentage required (default: 80)')
    args = parser.parse_args()
    
    console.print("[bold cyan]🔍 EOL RAG Context Documentation Validator[/bold cyan]\n")
    
    # Validate MkDocs config
    console.print("[blue]Validating MkDocs configuration...[/blue]")
    mkdocs_valid = validate_mkdocs_config()
    
    # Analyze docstring coverage
    console.print(f"\n[blue]Analyzing docstring coverage in {args.path}...[/blue]")
    stats, missing = analyze_directory(args.path)
    
    # Print report
    print_coverage_report(stats, missing, args.verbose)
    
    # Check if we meet minimum coverage
    coverage = calculate_coverage(stats)
    success = coverage >= args.min_coverage and mkdocs_valid
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()