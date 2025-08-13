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
  - Update progress in real-time
  - Mark tasks complete immediately
  - Track blockers actively
  ```

  **`.claude/plans/completed/CLAUDE.md`:**
  ```markdown
  # Completed State Rules
  - Extract lessons learned
  - Update context/ with discoveries
  - Archive for reference
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

---

## Phase 4: Root CLAUDE.md Enhancement

### 4.1 Update Project Root CLAUDE.md
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
  - Update task status immediately (never batch)
  - Mark completions in real-time
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

### What NOT to Migrate
- JIRA ticket integration (too specific)
- AWS/CDK patterns (not applicable)
- NX monorepo concepts (use Python/uv concepts instead)
- Complex parallel execution matrices (simplify)
- GitHub automation scripts (different tooling)

---

## Final Structure with Content Sources

```
.claude/
├── CLAUDE.md                        # ← Context Engineering, navigation
├── context/                         
│   ├── CLAUDE.md                   # ← Verification, read-before-edit
│   ├── planning-methodology.md      # ← Checkbox format, phases
│   ├── ai-development-patterns.md   # ← Error recovery, quality gates
│   ├── lessons-learned.md          # ← Anti-patterns, pitfalls
│   ├── quality-gates.md            # ← Performance thresholds
│   ├── rag/
│   │   ├── CLAUDE.md              # ← RAG metrics, targets
│   │   └── *.md                   # RAG-specific patterns
│   ├── redis/
│   │   ├── CLAUDE.md              # ← Redis best practices
│   │   └── *.md                   # Redis patterns
│   └── python/
│       ├── CLAUDE.md              # ← Python standards
│       └── *.md                   # Python patterns
├── plans/
│   ├── CLAUDE.md                   # ← State lifecycle, status tracking
│   ├── draft/
│   │   └── CLAUDE.md              # ← Flexibility focus
│   ├── ready/
│   │   └── CLAUDE.md              # ← Validation focus
│   ├── pending/
│   │   └── CLAUDE.md              # ← Real-time updates
│   └── completed/
│       └── CLAUDE.md              # ← Lessons extraction
└── settings.local.json

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

## Notes

This migration plan specifically maps which MillTechFX concepts go to which CLAUDE.md files, ensuring:
1. **Appropriate distribution** of rules and context
2. **Adaptation** to Python/RAG/Redis environment
3. **Hierarchical organization** leveraging nested CLAUDE.md
4. **Clear boundaries** between different contexts

The nested CLAUDE.md architecture ensures AI behavior adapts appropriately to each directory's purpose while maintaining the valuable patterns from MillTechFX.