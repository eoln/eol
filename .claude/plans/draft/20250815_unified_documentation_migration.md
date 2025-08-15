# Unified Documentation Migration Plan

**Plan ID**: 20250815_unified_documentation_migration  
**Created**: 2025-08-15  
**Status**: draft  
**Confidence**: 9/10  
**Estimated Duration**: 2-3 hours  
**Branch**: feat/unified-docs  

## Objective

Migrate from package-specific documentation (eol-rag-context) to a unified monorepo documentation site at the root level, improving maintainability and enabling proper pre-commit hooks.

## Current State Analysis

### Current Structure
```
eol/
├── packages/
│   └── eol-rag-context/
│       ├── mkdocs.yml           # Package-specific MkDocs config
│       ├── docs/                 # Package documentation
│       │   ├── api-reference/
│       │   ├── getting-started/
│       │   ├── user-guide/
│       │   └── examples/
│       ├── scripts/
│       │   └── validate_docs.py  # Documentation validator
│       └── .pre-commit-config.yaml  # Package-level hooks
└── (no root-level docs)
```

### Issues with Current Setup
1. Pre-commit hooks fail when run from root directory
2. Documentation is fragmented across packages
3. No unified view of the entire EOL framework
4. Duplicate configuration and tooling
5. Harder to maintain consistent documentation standards

## Proposed Structure

### Target Structure
```
eol/
├── mkdocs.yml                   # Root-level unified config
├── docs/                         # Unified documentation
│   ├── index.md                  # Main landing page
│   ├── getting-started/
│   │   ├── installation.md
│   │   └── quickstart.md
│   ├── packages/                 # Package-specific docs
│   │   ├── eol-rag-context/
│   │   │   ├── index.md
│   │   │   ├── api-reference/
│   │   │   ├── user-guide/
│   │   │   └── examples/
│   │   ├── eol-core/            # Future package
│   │   ├── eol-cli/             # Future package
│   │   └── eol-sdk/             # Future package
│   ├── architecture/
│   │   ├── overview.md
│   │   ├── monorepo-structure.md
│   │   └── design-decisions.md
│   ├── development/
│   │   ├── contributing.md
│   │   ├── testing.md
│   │   └── ci-cd.md
│   └── api-reference/           # Combined API docs
├── scripts/
│   └── validate_docs.py         # Root-level validator
└── .pre-commit-config.yaml      # Root-level hooks
```

## Implementation Plan

### Phase 1: Setup Root Documentation Infrastructure

#### Task 1.1: Create Root MkDocs Configuration
```yaml
# /mkdocs.yml
site_name: EOL Framework
site_description: End-of-Line AI Framework - Production-ready RAG infrastructure
site_url: https://eoln.github.io/eol/
repo_url: https://github.com/eoln/eol
repo_name: eoln/eol

theme:
  name: material
  custom_dir: docs/overrides
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: blue
      accent: blue
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: blue
      accent: blue
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - announce.dismiss
    - content.action.edit
    - content.action.view
    - content.code.annotate
    - content.code.copy
    - content.tooltips
    - navigation.footer
    - navigation.indexes
    - navigation.instant
    - navigation.sections
    - navigation.tabs
    - navigation.tabs.sticky
    - navigation.top
    - navigation.tracking
    - search.highlight
    - search.share
    - search.suggest
    - toc.follow
    - toc.integrate

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [packages/eol-rag-context/src, packages/eol-core/src, packages/eol-cli/src]
          options:
            show_source: true
            show_symbol_type_heading: true
            show_symbol_type_toc: true
  - awesome-pages
  - mike:  # For versioning
      version_selector: true
      css_dir: css
      javascript_dir: js
      canonical_version: latest

markdown_extensions:
  - admonition
  - codehilite
  - pymdownx.superfences
  - pymdownx.tabbed:
      alternate_style: true
  - pymdownx.details
  - pymdownx.emoji
  - pymdownx.tasklist
  - toc:
      permalink: true

nav:
  - Home: index.md
  - Getting Started:
      - Installation: getting-started/installation.md
      - Quick Start: getting-started/quickstart.md
      - Configuration: getting-started/configuration.md
  - Packages:
      - Overview: packages/index.md
      - EOL RAG Context:
          - Introduction: packages/eol-rag-context/index.md
          - User Guide: packages/eol-rag-context/user-guide/
          - API Reference: packages/eol-rag-context/api-reference/
          - Examples: packages/eol-rag-context/examples/
      - EOL Core: packages/eol-core/index.md
      - EOL CLI: packages/eol-cli/index.md
      - EOL SDK: packages/eol-sdk/index.md
  - Architecture:
      - Overview: architecture/overview.md
      - Monorepo Structure: architecture/monorepo-structure.md
      - Design Decisions: architecture/design-decisions.md
  - Development:
      - Contributing: development/contributing.md
      - Testing: development/testing.md
      - CI/CD: development/ci-cd.md
  - API Reference: api-reference/
```

#### Task 1.2: Create Root Documentation Structure
```bash
# Create directories
mkdir -p docs/{getting-started,packages,architecture,development,api-reference,overrides}
mkdir -p docs/packages/{eol-rag-context,eol-core,eol-cli,eol-sdk}

# Move validate_docs.py to root scripts
mkdir -p scripts
cp packages/eol-rag-context/scripts/validate_docs.py scripts/
# Update paths in validate_docs.py
```

### Phase 2: Migrate Package Documentation

#### Task 2.1: Move eol-rag-context Docs
```bash
# Copy existing docs
cp -r packages/eol-rag-context/docs/* docs/packages/eol-rag-context/

# Update internal links
# FROM: /api-reference/config
# TO: /packages/eol-rag-context/api-reference/config
```

#### Task 2.2: Create Landing Pages
- Main index.md with framework overview
- Package index pages with navigation
- Architecture documentation
- Development guides

### Phase 3: Update Build System

#### Task 3.1: Update Pre-commit Hooks
```yaml
# /.pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-docs
        name: Validate Documentation Coverage
        entry: python scripts/validate_docs.py
        language: system
        pass_filenames: false
        always_run: true
        
      - id: mkdocs-build
        name: Test MkDocs Build
        entry: mkdocs build --strict --quiet
        language: system
        pass_filenames: false
        files: ^(docs/|mkdocs.yml|packages/.*/src/.*\.py)$
```

#### Task 3.2: Update GitHub Actions
```yaml
# /.github/workflows/docs.yml
name: Documentation

on:
  push:
    branches: [main]
    paths:
      - 'docs/**'
      - 'mkdocs.yml'
      - 'packages/*/src/**/*.py'
  pull_request:
    paths:
      - 'docs/**'
      - 'mkdocs.yml'
      - 'packages/*/src/**/*.py'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install mkdocs mkdocs-material mkdocstrings[python] \
                      mkdocs-awesome-pages-plugin mike
      
      - name: Build documentation
        run: mkdocs build --strict
      
      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main'
        run: |
          mike deploy --push --update-aliases latest
```

### Phase 4: Testing and Validation

#### Task 4.1: Test Documentation Build
```bash
# Install dependencies
pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-awesome-pages-plugin

# Build locally
mkdocs build --strict

# Serve locally
mkdocs serve

# Test pre-commit hooks
pre-commit run --all-files
```

#### Task 4.2: Validate Links and Coverage
```bash
# Run documentation validator
python scripts/validate_docs.py

# Check for broken links
mkdocs build --strict 2>&1 | grep -E "WARNING|ERROR"
```

## Benefits of Unified Documentation

### 1. **Improved Developer Experience**
- Single documentation site for entire framework
- Better navigation between packages
- Unified search across all documentation

### 2. **Better Pre-commit Integration**
- Hooks work from any directory
- Consistent validation across all packages
- Single source of truth for documentation standards

### 3. **Easier Maintenance**
- One MkDocs configuration to maintain
- Shared themes and plugins
- Centralized documentation scripts

### 4. **Enhanced CI/CD**
- Single deployment pipeline
- Versioned documentation with mike
- Automated API documentation generation

### 5. **Scalability**
- Easy to add new packages
- Consistent structure for all packages
- Shared documentation components

## Migration Checklist

- [ ] Create root mkdocs.yml configuration
- [ ] Set up root docs/ directory structure
- [ ] Move eol-rag-context documentation
- [ ] Update all internal documentation links
- [ ] Move validate_docs.py to root scripts/
- [ ] Update pre-commit hooks configuration
- [ ] Create root .pre-commit-config.yaml
- [ ] Update GitHub Actions workflows
- [ ] Test local documentation build
- [ ] Test pre-commit hooks from root
- [ ] Create main landing pages
- [ ] Add architecture documentation
- [ ] Add development guides
- [ ] Test deployment to GitHub Pages
- [ ] Update README.md with new docs location
- [ ] Remove package-level mkdocs.yml
- [ ] Clean up old documentation files

## Risk Mitigation

### Risk 1: Broken Links
**Mitigation**: Use automated link checking in CI/CD

### Risk 2: Lost Documentation
**Mitigation**: Keep backup of current docs before migration

### Risk 3: CI/CD Failures
**Mitigation**: Test workflows in feature branch first

### Risk 4: Pre-commit Hook Issues
**Mitigation**: Test hooks locally before committing

## Success Criteria

- ✅ Documentation builds without errors
- ✅ All pre-commit hooks pass from root directory
- ✅ Documentation is accessible at https://eoln.github.io/eol/
- ✅ All internal links work correctly
- ✅ API documentation generates automatically
- ✅ Search works across all packages
- ✅ Documentation coverage > 95%

## Next Steps

1. Review and approve this plan
2. Create feat/unified-docs branch
3. Execute Phase 1: Setup infrastructure
4. Execute Phase 2: Migrate documentation
5. Execute Phase 3: Update build system
6. Execute Phase 4: Test and validate
7. Create PR for review
8. Deploy to GitHub Pages

---

*This plan unifies documentation for better maintainability and developer experience.*