# Product Requirements Prompt (PRP) Commands

Systematic planning methodology for creating comprehensive implementation blueprints for any software feature development using context engineering principles.

## Overview

PRP commands implement the Product Requirements Prompt methodology - a context engineering approach that provides AI coding assistants with comprehensive implementation blueprints for production-ready software development.

**Core Principle**: PRP = PRD + Curated Codebase Intelligence + Agent Runbook

**Key Philosophy**: Context Engineering is the systematic process of giving AI coding assistants a comprehensive mission briefing - complete with requirements, code examples, technical constraints and best practices - before writing code.

## PRP Context Engineering Workflow

1. **Analyze** - Extract patterns, dependencies, and technical context from codebase
2. **Create** - Generate comprehensive implementation blueprint with curated intelligence
3. **Review** - Validate plan accuracy and completeness against current patterns
4. **Execute** - Systematic implementation with self-correction and validation gates
5. **Validate** - Ensure production-ready code quality and performance targets

## Command Suite

| Command | Purpose | Usage |
|---------|---------|-------|
| `analyze` | Extract patterns from codebase for planning | `/prp:analyze "domain" "technology"` |
| `create` | Generate comprehensive implementation blueprint | `/prp:create "feature description"` |
| `execute` | Execute PRP plan with systematic tracking | `/prp:execute .claude/plans/prp-file.md` |
| `validate` | Validate PRP completeness and feasibility | `/prp:validate .claude/plans/prp-file.md` |
| `review` | Review and update existing PRPs | `/prp:review .claude/plans/prp-file.md` |

## PRP Methodology

PRP commands implement the structured planning methodology documented in [`.claude/context/planning-methodology.md`](../../context/planning-methodology.md).

### Context Engineering Implementation

PRP commands generate comprehensive implementation blueprints following context engineering principles:

- **System Context**: Global project rules and conventions from `.claude/CLAUDE.md`
- **Domain Context**: Technology-specific patterns from `.claude/context/`
- **Task Context**: Feature requirements with technical constraints
- **Implementation Blueprint**: Phase-based tasks with validation gates (see [Phase Structure](../../context/planning-methodology.md#phase-based-development))
- **Validation Framework**: Quality gates, testing, and success metrics (see [Defining Success](../../context/planning-methodology.md#success-metrics))

### Systematic Execution

- Phase-based implementation ([Phase Guidelines](../../context/planning-methodology.md#phase-guidelines))
- Real-time task tracking with checkbox format
- Git branch strategy ([Branch Lifecycle](../../context/planning-methodology.md#git-branch-strategy))
- Quality gates at each phase boundary
- Performance validation against defined metrics

## Best Practices

1. Always analyze before creating PRPs
2. Use systematic task tracking
3. Validate at each phase
4. Document lessons learned
5. Update context with discoveries
