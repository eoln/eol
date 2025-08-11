# EOL Monorepo Dependency Management

## Overview

The EOL project uses **UV** (by Astral) as its primary dependency management tool, implementing a modern Python monorepo pattern with centralized dependency management and workspace support.

## Quick Start

### Initial Setup
```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up the workspace
./scripts/setup-workspace.sh

# Activate virtual environment
source .venv/bin/activate
```

### Daily Workflow
```bash
# Sync all dependencies
./scripts/sync-deps.sh

# Check for issues
./scripts/check-deps.sh

# Run all tests
./scripts/test-all.sh

# Update dependencies
./scripts/update-deps.sh patch  # or minor, major, all
```

## Project Structure

```
eol/
├── pyproject.toml              # Root workspace configuration
├── uv.toml                     # UV-specific settings
├── requirements/
│   ├── base.txt               # Shared core dependencies
│   ├── dev.txt                # Development dependencies
│   └── constraints.txt        # Version constraints (lock file)
├── packages/
│   ├── eol-core/              # Core framework package
│   │   └── pyproject.toml    # Package configuration
│   ├── eol-cli/               # CLI package
│   │   └── pyproject.toml
│   └── eol-rag-context/       # RAG context MCP server
│       └── pyproject.toml
└── scripts/
    ├── setup-workspace.sh     # Initial setup
    ├── sync-deps.sh          # Sync dependencies
    ├── check-deps.sh         # Check for issues
    ├── update-deps.sh        # Update dependencies
    └── test-all.sh           # Run all tests
```

## Dependency Management Strategy

### 1. Three-Tier System

#### Tier 1: Shared Dependencies (`requirements/base.txt`)
Core dependencies used across multiple packages:
- `pydantic` - Data validation
- `typer` - CLI framework
- `rich` - Terminal formatting
- `numpy` - Numerical operations
- `pyyaml` - YAML processing

#### Tier 2: Package-Specific Dependencies
Each package's `pyproject.toml` declares its specific needs:
- `eol-core` - Minimal dependencies
- `eol-cli` - CLI-specific tools
- `eol-rag-context` - Redis, MCP, document processing

#### Tier 3: Development Dependencies (`requirements/dev.txt`)
Shared development tools:
- Testing: `pytest`, `pytest-cov`
- Linting: `black`, `ruff`, `mypy`
- Documentation: `mkdocs`, `mkdocstrings`
- Security: `pip-audit`, `safety`, `bandit`

### 2. Version Management

#### Constraints File (`requirements/constraints.txt`)
- Pins exact versions for reproducibility
- Updated via `./scripts/update-deps.sh`
- Shared across all packages

#### Update Strategy
```bash
# Update patch versions only (1.2.3 → 1.2.4)
./scripts/update-deps.sh patch

# Update minor versions (1.2.3 → 1.3.0)
./scripts/update-deps.sh minor

# Update major versions (1.2.3 → 2.0.0)
./scripts/update-deps.sh major

# Update all to latest
./scripts/update-deps.sh all

# Dry run (no changes)
./scripts/update-deps.sh patch true
```

### 3. Workspace Dependencies

Internal packages can depend on each other:
```toml
# In eol-cli/pyproject.toml
[tool.uv.sources]
eol-core = { workspace = true }
eol-rag-context = { workspace = true }
```

## UV Commands Reference

### Basic Commands
```bash
# Create virtual environment
uv venv

# Sync workspace dependencies
uv sync --all-packages

# Install specific package
uv pip install package-name

# Install with constraints
uv pip install -c requirements/constraints.txt package-name

# Show installed packages
uv pip list

# Show outdated packages
uv pip list --outdated

# Check for conflicts
uv pip check
```

### Workspace Commands
```bash
# Sync all workspace packages
uv sync --all-packages

# Sync specific package
uv sync --package eol-rag-context

# Install package in editable mode
uv pip install -e packages/eol-rag-context

# Compile requirements
uv pip compile requirements/base.txt -o requirements/base.lock
```

## Security Scanning

### Manual Security Checks
```bash
# Run all security checks
./scripts/check-deps.sh

# Individual tools
pip-audit --desc
safety check
bandit -r packages/*/src
```

### Automated Security (CI/CD)
- Runs on every PR affecting dependencies
- Weekly scheduled scans
- Creates security reports as artifacts
- Generates GitHub Security advisories

## CI/CD Integration

### GitHub Actions Workflows

#### `.github/workflows/dependencies.yml`
- **Triggers**: On dependency changes, weekly schedule
- **Jobs**:
  - `dependency-check`: Validates dependencies across Python versions
  - `security-scan`: Runs security audits
  - `update-constraints`: Auto-creates PRs for updates
  - `validate-workspace`: Ensures workspace integrity

### Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Troubleshooting

### Common Issues

#### 1. UV Not Found
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
# Add to PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

#### 2. Dependency Conflicts
```bash
# Check conflicts
uv pip check

# Show dependency tree
pipdeptree

# Reinstall with constraints
uv pip install -c requirements/constraints.txt -r requirements/base.txt
```

#### 3. Import Errors
```bash
# Ensure virtual environment is active
source .venv/bin/activate

# Reinstall in editable mode
uv pip install -e packages/eol-rag-context
```

#### 4. Version Mismatch
```bash
# Force sync with constraints
uv pip sync requirements/constraints.txt

# Rebuild virtual environment
rm -rf .venv
./scripts/setup-workspace.sh
```

## Best Practices

### DO:
✅ Always use UV for dependency management
✅ Keep constraints.txt updated
✅ Run security checks before merging
✅ Use workspace dependencies for internal packages
✅ Test across Python versions (3.11, 3.12)
✅ Document new dependencies in pyproject.toml

### DON'T:
❌ Use pip directly (use `uv pip` instead)
❌ Install without constraints in production
❌ Mix dependency management tools
❌ Ignore security warnings
❌ Update all dependencies at once without testing

## Adding New Dependencies

### To a Specific Package
```bash
cd packages/eol-rag-context
uv pip install new-package
# Add to pyproject.toml dependencies
# Update constraints
uv pip freeze | grep new-package >> ../../requirements/constraints.txt
```

### To Shared Dependencies
1. Add to `requirements/base.txt` or `requirements/dev.txt`
2. Add version to `requirements/constraints.txt`
3. Run `./scripts/sync-deps.sh`
4. Test all packages: `./scripts/test-all.sh`

## Migration from pip/poetry

### From requirements.txt
```bash
# Convert existing requirements
cat old-requirements.txt >> requirements/base.txt
# Install with UV
uv pip install -r requirements/base.txt
# Generate constraints
uv pip freeze > requirements/constraints.txt
```

### From Poetry
```bash
# Export from Poetry
poetry export -f requirements.txt > old-requirements.txt
# Follow requirements.txt migration above
```

## Performance Tips

1. **Use UV's cache**: UV caches packages globally
2. **Parallel installation**: UV installs in parallel by default
3. **Compile requirements**: Pre-compile for faster CI/CD
4. **Local package links**: Use workspace dependencies to avoid reinstalls

## Monitoring and Maintenance

### Weekly Tasks
- Review security scan results
- Check for outdated packages
- Update patch versions

### Monthly Tasks
- Update minor versions
- Review and clean unused dependencies
- Update development tools

### Quarterly Tasks
- Consider major version updates
- Audit dependency tree
- Review dependency strategy

## Resources

- [UV Documentation](https://docs.astral.sh/uv/)
- [Python Packaging Guide](https://packaging.python.org/)
- [PEP 517](https://peps.python.org/pep-0517/) - Build system specification
- [PEP 621](https://peps.python.org/pep-0621/) - Project metadata specification

## Support

For dependency-related issues:
1. Check this documentation
2. Run `./scripts/check-deps.sh` for diagnostics
3. Review CI/CD logs for automated checks
4. Open an issue with dependency report attached