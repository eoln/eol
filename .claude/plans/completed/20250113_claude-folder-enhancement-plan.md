# .claude Folder Enhancement Plan - EOL RAG Framework

## Overview

This plan enhances the `.claude/` folder structure leveraging nested CLAUDE.md files for directory-specific AI behavior customization. Each section specifies exactly what content migrates from MillTechFX to which CLAUDE.md file.

**Created**: 2025-01-12
**Status**: Ready
**Priority**: High
**Estimated Duration**: 1 week
**Scope**: `.claude/` folder only

---

## Phase 1: Nested CLAUDE.md Architecture with Content Migration

### 1.1 Root .claude/CLAUDE.md

- [ ] **Create `.claude/CLAUDE.md`**

  **Migrate from MillTechFX:**
  - Context Engineering principles (WRITE, SELECT, COMPRESS, ISOLATE strategies)
  - Directory structure overview
  - General navigation rules

  **Content:**

  ```markdown
  # .claude Directory Rules

  ## Context Engineering Principles
  - WRITE: Use plans/ for persistent task tracking
  - SELECT: Load context/ files based on task requirements
  - COMPRESS: Keep context window <80% utilization
  - ISOLATE: Respect directory boundaries

  ## Directory Navigation
  - Always check for nested CLAUDE.md when entering subdirectories
  - Subdirectory rules override parent rules
  - Load relevant context before ANY operation
  ```

### 1.2 Plans Directory Structure

- [ ] **Create `.claude/plans/CLAUDE.md`**

  **Migrate from MillTechFX:**
  - Plan state lifecycle (draft → ready → pending → completed)
  - Task status indicators
  - Real-time progress tracking requirements
  - Immediate completion marking rules

  **Content:**

  ```markdown
  # Plan Management Rules

  ## State Transitions
  - ALWAYS move files between state directories
  - Update task status immediately (no batching)
  - Mark completed tasks instantly

  ## Task Status Within Plans
  - [ ] Pending
  - [~] In Progress
  - [x] Completed
  - [!] Blocked
  ```

### 1.3 State-Specific CLAUDE.md Files

- [ ] **Create `.claude/plans/README.md`**

  ```markdown
  # Plans Directory Guide

  ## State Definitions
  - **draft/**: Initial ideas, exploration, incomplete
  - **ready/**: Validated, approved, ready to execute
  - **pending/**: Currently being executed
  - **completed/**: Finished, archived for reference

  ## Workflow
  draft → ready → pending → completed

  ## File Naming
  Use descriptive kebab-case: feature-name-plan.md
  ```

### 1.4 State Directory CLAUDE.md

- [ ] **Create state directory rules**

  **`.claude/plans/draft/CLAUDE.md`:**

  ```markdown
  # Draft State Rules
  - Exploration and flexibility encouraged
  - Incomplete sections acceptable
  - Focus on capturing ideas
  ```

  **`.claude/plans/ready/CLAUDE.md`:**

  ```markdown
  # Ready State Rules
  - Validate all requirements documented
  - Ensure success metrics defined
  - Check dependencies identified
  ```

  **`.claude/plans/pending/CLAUDE.md`:**

  ```markdown
  # Pending State Rules
  - Create dedicated git branch before starting execution
  - Branch naming: feat/[plan-name] or fix/[plan-name]
  - Update progress in real-time
  - Mark tasks complete immediately
  - Track blockers actively
  - Commit changes incrementally with clear messages
  ```

  **`.claude/plans/completed/CLAUDE.md`:**

  ```markdown
  # Completed State Rules
  - Create PR from feature branch to main
  - Extract lessons learned
  - Update context/ with discoveries
  - Archive for reference
  - Merge PR after review and approval
  ```

---

## Phase 2: Context Directory Migration

### 2.1 Context Root Rules

- [ ] **Create `.claude/context/CLAUDE.md`**

  **Migrate from MillTechFX:**
  - Context verification requirements
  - "Always read before edit" principle
  - Dynamic context discovery patterns

  **Content:**

  ```markdown
  # Context Directory Rules

  ## Core Principles
  - ALWAYS read existing files before making changes
  - Never assume - always verify
  - Update immediately with discoveries

  ## Usage Patterns
  - Load domain-specific subdirectories for specialized tasks
  - Check multiple contexts for complex operations
  - Keep examples from actual codebase only
  ```

### 2.2 Domain-Specific Context Organization

#### RAG Context

- [ ] **Create RAG-specific content files**
  - `document-processing.md` - Chunking strategies, metadata extraction
  - `vector-search.md` - Query optimization, hybrid search patterns
  - `semantic-caching.md` - Cache key generation, TTL strategies

#### RAG Context CLAUDE.md

- [ ] **Create `.claude/context/rag/CLAUDE.md`**

  **Content:**

  ```markdown
  # RAG Context Rules

  ## Performance Targets
  - Document indexing: >10 docs/sec
  - Vector search: <100ms latency
  - Cache hit rate: >31%

  ## Always Consider
  - Chunking strategy for content type
  - Embedding model consistency
  - Metadata design
  ```

#### Redis Context

- [ ] **Create Redis-specific content files**
  - `vector-operations.md` - Redis vector patterns, index design
  - `performance-tuning.md` - Memory optimization, query performance
  - `connection-management.md` - Connection pooling, pipeline operations

#### Redis Context CLAUDE.md

- [ ] **Create `.claude/context/redis/CLAUDE.md`**

  **Content:**

  ```markdown
  # Redis Context Rules

  ## Patterns to Follow
  - Always use connection pooling
  - Pipeline for batch operations
  - Implement retry logic

  ## Performance Focus
  - Monitor memory usage
  - Optimize key patterns
  - Track operation latency
  ```

#### Python Context

- [ ] **Create Python-specific content files**
  - `async-patterns.md` - Async/await best practices for RAG
  - `type-hints.md` - Type annotation patterns, mypy compliance
  - `testing-strategies.md` - Pytest patterns, fixtures, mocking

#### Python Context CLAUDE.md

- [ ] **Create `.claude/context/python/CLAUDE.md`**

  **Content:**

  ```markdown
  # Python Context Rules

  ## Code Standards
  - Type hints for all functions
  - Async/await for I/O
  - Dataclasses for structures

  ## Testing Requirements
  - Minimum 80% coverage
  - Test success and failure paths
  - Use pytest fixtures
  ```

---

## Phase 3: Core Context Files with MillTechFX Patterns

### 3.1 Planning Methodology

- [ ] **Create `.claude/context/planning-methodology.md`**

  **Migrate from MillTechFX:**
  - Structured checkbox format
  - Phase-based development
  - Parallel execution concepts (simplified)
  - Success metrics definitions
  - Risk mitigation patterns
  - Git branch strategy for plan execution

### 3.2 AI Development Patterns

- [ ] **Create `.claude/context/ai-development-patterns.md`**

  **Migrate from MillTechFX:**
  - Context verification workflow
  - Error recovery patterns
  - Quality gate concepts
  - Anti-patterns from experience

### 3.3 Lessons Learned

- [ ] **Create `.claude/context/lessons-learned.md`**

  **Migrate from MillTechFX:**
  - Linting issue resolution patterns
  - User correction patterns
  - Tool usage best practices
  - Common pitfalls and solutions

### 3.4 Quality Gates

- [ ] **Create `.claude/context/quality-gates.md`**

  **Merge from existing GitHub Actions:**
  - Consolidate quality checks from `eol-rag-context-quality-gate.yml`
  - Performance thresholds (document indexing >10 docs/sec, vector search <100ms)
  - Coverage requirements (>80% for Python)
  - Security gates (Trivy, safety, bandit)
  - Code quality (Black, isort, flake8)
  - Testing requirements (unit, integration, performance)

---

## Phase 4: PRP Commands Migration

### 4.1 Commands Directory Structure

- [ ] **Create `.claude/commands/` directory**

  **Purpose:**
  - Store reusable command patterns and workflows
  - Provide quick access to common PR preparation tasks
  - Standardize development workflows across the project

### 4.2 Commands Root CLAUDE.md

- [ ] **Create `.claude/commands/CLAUDE.md`**

  **Content:**

  ```markdown
  # Commands Directory Rules

  ## Purpose
  - Store reusable command patterns for common tasks
  - Provide PR preparation (PRP) workflows
  - Standardize development operations

  ## Command Types
  - **prp/**: Pull Request Preparation commands
  - **quality/**: Quality check commands
  - **deploy/**: Deployment commands (future)

  ## Usage Pattern
  - Load command files as needed for specific workflows
  - Commands should be executable snippets or clear instructions
  - Include context and prerequisites for each command
  ```

### 4.3 PRP Commands Migration

- [ ] **Create `.claude/commands/prp/` directory**

#### PRP Directory CLAUDE.md

- [ ] **Create `.claude/commands/prp/CLAUDE.md`**

  **Content:**

  ```markdown
  # Pull Request Preparation (PRP) Commands

  ## Workflow
  1. Run quality checks
  2. Update documentation
  3. Verify tests pass
  4. Create PR with proper description

  ## Command Naming
  - Use descriptive names: `check-quality.md`, `prepare-pr.md`
  - Include prerequisites and expected outcomes
  ```

#### Core PRP Commands

- [ ] **Create `.claude/commands/prp/check-quality.md`**

  **Migrate from MillTechFX:**

  ```markdown
  # Quality Check Command

  ## Prerequisites
  - Python environment activated
  - Redis running (for integration tests)

  ## Command Sequence
  ```bash
  # Format code
  python -m black src/ tests/
  python -m isort src/ tests/

  # Lint
  python -m flake8 src/ tests/
  python -m mypy src/

  # Security checks
  safety check
  bandit -r src/

  # Run tests with coverage
  python -m pytest tests/ --cov=eol.rag_context --cov-report=term --cov-report=html
  ```

  ## Success Criteria

  - All formatters pass without changes
  - No linting errors
  - Coverage > 80%
  - No security vulnerabilities

  ```

- [ ] **Create `.claude/commands/prp/prepare-pr.md`**

  **Content:**

  ```markdown
  # Prepare Pull Request Command

  ## Prerequisites
  - All quality checks passed
  - Working on feature branch
  - Changes committed locally

  ## Command Sequence
  ```bash
  # Ensure branch is up to date
  git fetch origin main
  git rebase origin/main

  # Run final quality check
  python -m pytest tests/ -q

  # Push branch
  git push -u origin $(git branch --show-current)

  # Create PR with template
  gh pr create \
    --title "[Type] Brief description" \
    --body-file .github/pull_request_template.md \
    --base main
  ```

  ## PR Description Template

  - Summary of changes
  - Test coverage report
  - Performance impact (if any)
  - Breaking changes (if any)

  ```

- [ ] **Create `.claude/commands/prp/update-docs.md`**

  **Content:**

  ```markdown
  # Documentation Update Command

  ## Prerequisites
  - Code changes complete
  - API changes identified

  ## Command Sequence
  ```bash
  # Generate API docs
  python -m pydoc -w src/eol/rag_context

  # Update mkdocs if needed
  mkdocs build --strict

  # Check for broken links
  mkdocs serve --dev-addr=127.0.0.1:8000
  ```

  ## Documentation Checklist

  - [ ] Update docstrings for changed functions
  - [ ] Update README if API changed
  - [ ] Add examples for new features
  - [ ] Update CHANGELOG.md

  ```

- [ ] **Create `.claude/commands/prp/verify-integration.md`**

  **Content:**

  ```markdown
  # Integration Verification Command

  ## Prerequisites
  - Redis server running
  - Test data available

  ## Command Sequence
  ```bash
  # Start Redis if not running
  redis-server --daemonize yes

  # Run integration tests only
  python -m pytest tests/integration/ -v

  # Test with different Python versions
  uv run --python 3.11 pytest tests/
  uv run --python 3.12 pytest tests/

  # Performance benchmarks
  python -m pytest tests/benchmarks/ --benchmark-only
  ```

  ## Success Criteria

  - All integration tests pass
  - Performance within thresholds
  - No regression in benchmarks

  ```

### 4.4 Quality Commands

- [ ] **Create `.claude/commands/quality/` directory**

- [ ] **Create `.claude/commands/quality/quick-check.md`**

  **Content:**

  ```markdown
  # Quick Quality Check

  ## Purpose
  Fast validation before commits

  ## Command
  ```bash
  # One-liner for quick validation
  python -m black --check src/ && python -m flake8 src/ && python -m pytest tests/ -q
  ```

  ```

- [ ] **Create `.claude/commands/quality/full-validation.md`**

  **Content:**

  ```markdown
  # Full Validation Suite

  ## Purpose
  Complete validation before PR

  ## Command Sequence
  ```bash
  # Full quality gate simulation
  ./scripts/quality-gate.sh || {
    echo "Quality gate failed"
    exit 1
  }
  ```

  ```

---

## Phase 5: Root CLAUDE.md Enhancement

### 5.1 Update Project Root CLAUDE.md

- [ ] **Add to existing `/Users/eoln/Devel/eol/CLAUDE.md`**

  **Add these sections:**

  ```markdown
  ## Nested CLAUDE.md Architecture

  The .claude/ folder uses nested CLAUDE.md files for directory-specific behavior:
  - Each subdirectory may contain its own CLAUDE.md with local rules
  - Subdirectory rules override parent rules for that context
  - Always check for local CLAUDE.md when entering directories

  ## Context Engineering

  Four strategies for effective AI assistance:
  - **WRITE**: Persistent memory via .claude/plans/
  - **SELECT**: Dynamic context loading from .claude/context/
  - **COMPRESS**: Optimize context window usage (<80%)
  - **ISOLATE**: Respect directory and security boundaries

  ## Planning Methodology

  - Use structured plans in .claude/plans/
  - Follow state lifecycle: draft → ready → pending → completed
  - ALWAYS create dedicated git branch for plan execution (e.g., feat/plan-name)
  - Update task status immediately (never batch)
  - Mark completions in real-time
  - Create PR when plan moves to completed state

  ## Command Patterns

  - Load commands from .claude/commands/ for common workflows
  - PRP commands for pull request preparation
  - Quality commands for validation and checks
  - Commands include prerequisites and success criteria

  ## Python Environment

  - ALWAYS activate virtual environment before running scripts
  - Use `source .venv/bin/activate` or appropriate venv activation
  - Prefer `uv` for package management and environment handling
  - Never install packages globally - always use venv
  ```

---

## Migration Mapping Summary

### From MillTechFX to EOL

| MillTechFX Concept | EOL Location | Adaptation |
|-------------------|--------------|------------|
| Context Engineering (4 strategies) | `.claude/CLAUDE.md` | Keep core concepts |
| Plan state lifecycle | `.claude/plans/CLAUDE.md` | Simplify for Python/RAG |
| Task status indicators | `.claude/plans/CLAUDE.md` | Keep as-is |
| Real-time updates | `.claude/plans/pending/CLAUDE.md` | Emphasize strongly |
| Context verification | `.claude/context/CLAUDE.md` | Core principle |
| Quality gates | `.claude/context/quality-gates.md` | Adapt for Python/Redis |
| Error recovery | `.claude/context/ai-development-patterns.md` | Keep patterns |
| Anti-patterns | `.claude/context/lessons-learned.md` | Add RAG-specific |
| Performance metrics | `.claude/context/rag/CLAUDE.md` | RAG-specific targets |
| PRP commands | `.claude/commands/prp/` | Adapt for Python/uv tooling |
| Quality workflows | `.claude/commands/quality/` | Python-specific checks |

### What NOT to Migrate

- JIRA ticket integration (too specific)
- AWS/CDK patterns (not applicable)
- NX monorepo concepts (use Python/uv concepts instead)
- Complex parallel execution matrices (simplify)
- GitHub automation scripts (migrate as command patterns instead)
- Build system specifics (NX vs uv workspace differences)

---

## Final Structure with Content Sources

```
.claude/
├── CLAUDE.md                        # ← Context Engineering, navigation
├── context/
│   ├── CLAUDE.md                   # ← Verification, read-before-edit
│   ├── planning-methodology.md      # ← Checkbox format, phases
│   ├── ai-development-patterns.md   # ← Error recovery, patterns
│   ├── lessons-learned.md          # ← Anti-patterns, pitfalls
│   ├── quality-gates.md            # ← Merged from GitHub Actions workflow
│   ├── rag/
│   │   ├── CLAUDE.md              # ← RAG metrics, targets
│   │   ├── document-processing.md
│   │   ├── vector-search.md
│   │   └── semantic-caching.md
│   ├── redis/
│   │   ├── CLAUDE.md              # ← Redis best practices
│   │   ├── vector-operations.md
│   │   ├── performance-tuning.md
│   │   └── connection-management.md
│   └── python/
│       ├── CLAUDE.md              # ← Python standards
│       ├── async-patterns.md
│       ├── type-hints.md
│       └── testing-strategies.md
├── commands/                        # ← Reusable command patterns
│   ├── CLAUDE.md                   # ← Command usage rules
│   ├── prp/                        # ← Pull Request Preparation
│   │   ├── CLAUDE.md              # ← PRP workflow
│   │   ├── check-quality.md       # ← Quality checks
│   │   ├── prepare-pr.md          # ← PR creation
│   │   ├── update-docs.md         # ← Documentation
│   │   └── verify-integration.md  # ← Integration tests
│   └── quality/                    # ← Quality commands
│       ├── quick-check.md         # ← Fast validation
│       └── full-validation.md     # ← Complete checks
├── plans/
│   ├── README.md                   # ← State definitions, workflow
│   ├── CLAUDE.md                   # ← State lifecycle, status tracking
│   ├── draft/
│   │   └── CLAUDE.md              # ← Flexibility focus
│   ├── ready/
│   │   └── CLAUDE.md              # ← Validation focus
│   ├── pending/
│   │   └── CLAUDE.md              # ← Real-time updates
│   └── completed/
│       └── CLAUDE.md              # ← Lessons extraction
└── settings.local.json              # ← Existing MCP settings

Project Root:
├── CLAUDE.md                        # ← Add nested rules section
```

---

## Success Metrics

### Migration Completeness

- [ ] All specified content migrated to correct CLAUDE.md files
- [ ] Adaptations made for Python/RAG context
- [ ] No inappropriate concepts carried over

### Functionality

- [ ] Nested rules work as expected
- [ ] Context loading follows hierarchy
- [ ] Plan state management functions

### AI Behavior

- [ ] Respects directory-specific rules
- [ ] Maintains real-time progress tracking
- [ ] Applies appropriate context for tasks

---

## Implementation Notes

### Quality Gates Integration

The existing `eol-rag-context-quality-gate.yml` GitHub Action provides comprehensive quality checks:

- **Code Quality**: Black formatting, isort imports, flake8 linting
- **Testing**: Unit tests (Python 3.11, 3.12), integration tests with Redis
- **Coverage**: 80% threshold with detailed reporting
- **Security**: Trivy scanning, safety dependency checks, bandit analysis
- **Performance**: Benchmark tests for RAG operations

These checks will be documented in `.claude/context/quality-gates.md` as the unified quality standard.

### PRP Commands Migration

The PRP (Pull Request Preparation) commands from MillTechFX are being migrated to `.claude/commands/prp/` with adaptations for Python/uv tooling:

- **Command Structure**: Each command file includes prerequisites, command sequences, and success criteria
- **Python-Specific**: Commands adapted for Python tools (black, flake8, pytest, mypy) instead of Node.js tools
- **Workflow Integration**: Commands can be loaded on-demand for PR preparation workflows
- **Quality Focus**: Emphasizes quality gates that match the GitHub Actions workflow

Key differences from MillTechFX:

- Uses `uv` instead of `npm/nx` for package management
- Python-specific linting and formatting tools
- Integration with Redis for RAG-specific testing
- Simplified command structure without complex matrices

### Settings.local.json

The existing `settings.local.json` file contains MCP server configuration and should remain unchanged. It's not part of the migration but listed in the structure for completeness.

## Notes

This migration plan specifically maps which MillTechFX concepts go to which CLAUDE.md files, ensuring:

1. **Appropriate distribution** of rules and context
2. **Adaptation** to Python/RAG/Redis environment
3. **Hierarchical organization** leveraging nested CLAUDE.md
4. **Clear boundaries** between different contexts

The nested CLAUDE.md architecture ensures AI behavior adapts appropriately to each directory's purpose while maintaining the valuable patterns from MillTechFX.
