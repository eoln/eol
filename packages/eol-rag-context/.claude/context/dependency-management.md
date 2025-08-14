# Python Monorepo Dependency Management Patterns

## Current State Analysis

The EOL project is structured as a Python monorepo with multiple packages:

- `packages/eol-core/` - Core framework engine
- `packages/eol-cli/` - Command-line interface
- `packages/eol-rag-context/` - RAG context MCP server (current package)

Currently, each package manages its own dependencies independently:

- Individual `pyproject.toml` per package
- Separate `requirements.txt` and `requirements-dev.txt` files
- No root-level dependency coordination

## Recommended Patterns for Python Monorepos

### 1. **Hybrid Approach (Recommended for EOL)**

This approach balances flexibility with consistency:

```
eol/
├── pyproject.toml              # Root workspace configuration
├── requirements/
│   ├── base.txt               # Shared core dependencies
│   ├── dev.txt                # Shared development tools
│   └── constraints.txt        # Version constraints for all packages
├── packages/
│   ├── eol-core/
│   │   ├── pyproject.toml    # Package-specific config
│   │   └── requirements.txt  # Additional deps (inherits base)
│   ├── eol-cli/
│   │   ├── pyproject.toml
│   │   └── requirements.txt
│   └── eol-rag-context/
│       ├── pyproject.toml
│       └── requirements.txt
```

**Benefits:**

- ✅ Centralized version control for common dependencies
- ✅ Package-specific flexibility for unique requirements
- ✅ Easier security updates across all packages
- ✅ Consistent development environment

**Implementation:**

```toml
# Root pyproject.toml
[tool.uv.workspace]
members = ["packages/*"]

[tool.uv.sources]
eol-core = { workspace = true }
eol-cli = { workspace = true }
```

### 2. **Tool-Specific Approaches**

#### **UV (Modern Python Package Manager)**

UV is particularly well-suited for monorepos with its workspace feature:

```bash
# Initialize workspace
uv init --workspace

# Add workspace members
uv workspace add packages/eol-core
uv workspace add packages/eol-cli
uv workspace add packages/eol-rag-context

# Install all dependencies
uv sync

# Add dependency to specific package
cd packages/eol-rag-context
uv add redis
```

**Advantages:**

- Built-in workspace support
- Fast dependency resolution
- Lock file for reproducibility
- Compatible with pip

#### **Poetry with Path Dependencies**

```toml
# packages/eol-rag-context/pyproject.toml
[tool.poetry.dependencies]
python = "^3.11"
eol-core = {path = "../eol-core", develop = true}
redis = "^5.0.0"
```

**Advantages:**

- Mature tooling
- Good IDE support
- Automatic virtual environment management

#### **Pants Build System**

For large-scale monorepos:

```python
# BUILD file in package root
python_sources(
    dependencies=[
        "//packages/eol-core:lib",
    ],
)

python_requirements(
    name="reqs",
    source="requirements.txt",
)
```

**Advantages:**

- Fine-grained caching
- Parallel builds
- Advanced dependency analysis

### 3. **Dependency Organization Patterns**

#### **Pattern 1: Centralized Dependencies (Simple)**

All dependencies in root, packages only declare what they use:

```
# /requirements/base.txt
redis>=5.0.0,<6.0.0
pydantic>=2.0.0,<3.0.0
fastmcp>=0.5.0,<1.0.0

# /packages/eol-rag-context/requirements.txt
-r ../../requirements/base.txt
# Package-specific additions
sentence-transformers>=2.2.0
```

**When to use:** Small to medium monorepos with high dependency overlap

#### **Pattern 2: Layered Dependencies (Scalable)**

```
requirements/
├── core.txt          # Absolute essentials (pydantic, typing-extensions)
├── data.txt          # Data layer (redis, postgresql, mongodb)
├── web.txt           # Web framework (fastapi, django)
├── ml.txt            # ML libraries (torch, transformers)
├── dev.txt           # Development tools
└── constraints.txt   # Version pins for all
```

**When to use:** Large monorepos with diverse package types

#### **Pattern 3: Package Groups (Domain-Driven)**

```
requirements/
├── framework/        # Core framework deps
│   ├── base.txt
│   └── plugins.txt
├── services/         # Service-specific deps
│   ├── api.txt
│   └── workers.txt
└── tools/           # CLI and development tools
    ├── cli.txt
    └── dev.txt
```

**When to use:** Domain-driven design with clear boundaries

### 4. **Version Management Strategies**

#### **Constraint Files**

Use a central constraints file to ensure version consistency:

```txt
# requirements/constraints.txt
redis==5.1.0
pydantic==2.5.3
numpy==1.24.0
```

```bash
# Install with constraints
pip install -r requirements.txt -c requirements/constraints.txt
```

#### **Version Ranges**

Define acceptable version ranges centrally:

```txt
# requirements/base.txt
redis>=5.0.0,<6.0.0  # Major version pinning
pydantic~=2.5.0      # Compatible release
numpy==1.24.0        # Exact version for stability
```

### 5. **Dependency Resolution Workflow**

#### **Development Workflow**

```bash
# 1. Add new dependency to package
cd packages/eol-rag-context
echo "new-package>=1.0.0" >> requirements.txt

# 2. Update constraints if needed
echo "new-package==1.2.3" >> ../../requirements/constraints.txt

# 3. Reinstall with constraints
pip install -r requirements.txt -c ../../requirements/constraints.txt

# 4. Test across all packages
./scripts/test_all_packages.sh
```

#### **CI/CD Integration**

```yaml
# .github/workflows/deps.yml
name: Dependency Management
on:
  pull_request:
    paths:
      - '**/requirements*.txt'
      - '**/pyproject.toml'

jobs:
  check-deps:
    steps:
      - name: Check for conflicts
        run: |
          pip-compile requirements/base.txt -o requirements/locked.txt
          pip check

      - name: Security audit
        run: |
          pip-audit -r requirements/locked.txt
          safety check
```

### 6. **Migration Path for EOL Project**

#### **Phase 1: Centralize Common Dependencies**

```bash
# Create central requirements
mkdir -p requirements
cat packages/*/requirements.txt | sort -u > requirements/analysis.txt

# Identify common deps (appear in 2+ packages)
# Move to requirements/base.txt
```

#### **Phase 2: Implement Constraints**

```bash
# Create constraints file
pip freeze > requirements/constraints.txt

# Update package requirements to use constraints
echo "-c ../../requirements/constraints.txt" >> packages/eol-rag-context/requirements.txt
```

#### **Phase 3: Adopt UV Workspaces**

```toml
# Root pyproject.toml
[project]
name = "eol-monorepo"
version = "0.1.0"

[tool.uv.workspace]
members = ["packages/*"]

[tool.uv]
constraint-dependencies = [
    "redis>=5.0.0,<6.0.0",
    "pydantic>=2.0.0,<3.0.0",
]
```

### 7. **Best Practices**

#### **DO:**

- ✅ Use constraint files for version consistency
- ✅ Pin major versions, allow minor updates
- ✅ Centralize common dependencies
- ✅ Use virtual environments per package for isolation
- ✅ Automate dependency updates with Dependabot
- ✅ Run security audits in CI/CD
- ✅ Document dependency decisions

#### **DON'T:**

- ❌ Duplicate version specifications across packages
- ❌ Use unpinned dependencies in production
- ❌ Mix dependency management tools
- ❌ Ignore transitive dependency conflicts
- ❌ Update all dependencies at once

### 8. **Tooling Recommendations**

#### **Essential Tools:**

```bash
# UV - Modern package manager
pip install uv

# pip-tools - Dependency compilation
pip install pip-tools

# pip-audit - Security scanning
pip install pip-audit

# pipdeptree - Dependency visualization
pip install pipdeptree
```

#### **Helper Scripts:**

```bash
#!/bin/bash
# scripts/sync-deps.sh

# Sync all package dependencies
for package in packages/*; do
    echo "Syncing $package..."
    cd $package
    pip-sync requirements.txt -c ../../requirements/constraints.txt
    cd ../..
done
```

### 9. **Common Issues and Solutions**

#### **Issue: Version Conflicts**

**Solution:** Use constraint files and test resolution in CI

#### **Issue: Slow Installation**

**Solution:** Use UV or implement caching layers

#### **Issue: Different Python Versions**

**Solution:** Use pyenv + virtual environments per package

#### **Issue: Private Dependencies**

**Solution:** Use index-url configuration or path dependencies

### 10. **Decision Matrix**

| Approach | Team Size | Packages | Complexity | Recommendation |
|----------|-----------|----------|------------|----------------|
| Individual deps | 1-3 | 2-5 | Low | Current approach OK |
| Centralized base | 3-10 | 5-15 | Medium | **Recommended for EOL** |
| UV Workspaces | 5-20 | 10-50 | Medium-High | Future migration |
| Pants/Bazel | 20+ | 50+ | High | Enterprise scale |

## Recommended Next Steps for EOL

1. **Immediate (Phase 1):**
   - Create `requirements/` directory
   - Extract common dependencies to `requirements/base.txt`
   - Add `requirements/constraints.txt` with current versions

2. **Short-term (Phase 2):**
   - Update CI/CD to use constraint files
   - Add dependency security scanning
   - Implement automated updates

3. **Long-term (Phase 3):**
   - Migrate to UV workspaces
   - Implement path dependencies for internal packages
   - Add dependency graph visualization

## Example Implementation for EOL

```bash
# Directory structure
eol/
├── requirements/
│   ├── base.txt               # Common to all packages
│   ├── dev.txt                # Development tools
│   ├── constraints.txt        # Version locks
│   └── README.md              # Dependency docs
├── packages/
│   ├── eol-core/
│   │   └── requirements.txt   # -r ../../requirements/base.txt + specific
│   ├── eol-cli/
│   │   └── requirements.txt   # -r ../../requirements/base.txt + specific
│   └── eol-rag-context/
│       └── requirements.txt   # -r ../../requirements/base.txt + specific
└── scripts/
    ├── sync-deps.sh           # Sync all dependencies
    ├── check-deps.sh          # Check for conflicts
    └── update-deps.sh         # Update dependencies
```

This approach provides the best balance of:

- **Simplicity**: Easy to understand and implement
- **Flexibility**: Packages can have unique dependencies
- **Maintainability**: Central version management
- **Scalability**: Can grow with the project
