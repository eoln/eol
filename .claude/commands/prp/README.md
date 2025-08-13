# PRP Commands - Product Requirements Prompt Planning Methodology

Systematic planning methodology for creating comprehensive implementation blueprints for any software feature development using context engineering principles.

## Overview

Product Requirements Prompt (PRP) commands implement a structured context engineering methodology that transforms feature requirements into detailed, actionable implementation blueprints. PRP goes beyond traditional Product Requirements Documents (PRDs) by providing AI coding assistants with comprehensive context including existing code patterns, technical constraints, and validation frameworks.

**Key Philosophy**: Context Engineering is the systematic process of giving AI coding assistants a comprehensive mission briefing - complete with requirements, code examples, technical constraints and best practices - before writing code.

## Command Suite

### Core Planning Commands

| Command | Purpose | Usage |
|---------|---------|-------|
| `analyze` | Extract patterns from codebase for planning | `/prp:analyze "domain" "technology"` |
| `create` | Generate comprehensive implementation blueprint | `/prp:create "feature description"` |
| `execute` | Execute PRP plan with systematic tracking | `/prp:execute .claude/plans/prp-file.md` |
| `validate` | Validate PRP completeness and feasibility | `/prp:validate .claude/plans/prp-file.md` |
| `review` | Review and update existing PRPs | `/prp:review .claude/plans/prp-file.md` |

### Quick Start Workflow

```bash
# 1. Analyze existing RAG patterns
/prp:analyze "document-processing" "vector-search"

# 2. Create implementation blueprint
/prp:create "enhanced semantic caching with TTL management"

# 3. Execute with systematic tracking
/prp:execute .claude/plans/prp-semantic-caching.md

# 4. Validate quality and completeness
/prp:validate .claude/plans/prp-semantic-caching.md --full
```

## PRP Methodology Focus

### Planning Philosophy
PRP commands implement the structured planning methodology detailed in [`.claude/context/planning-methodology.md`](../../context/planning-methodology.md).

PRP commands are specifically designed for the **planning and design phase** of development:
- **Not** for executing code quality checks (use `/quality:*` commands)
- **Not** for PR preparation (use `/project:prepare-pr`)
- **Not** for documentation updates (use `/project:update-docs`)
- **Focused** on creating, reviewing, and executing implementation blueprints

### What PRPs Do
1. **Analyze** existing codebase patterns and technical context
2. **Create** comprehensive implementation blueprints with curated intelligence
3. **Execute** plans with systematic validation and self-correction
4. **Validate** implementation quality and production readiness
5. **Review** and update plans to maintain accuracy and relevance

### Context Engineering Layers
Following the five-layer context engineering architecture:
- **System Context**: Global project rules and conventions
- **Domain Context**: Specific technology patterns (RAG, Redis, Python, etc.)
- **Task Context**: Feature-specific requirements and constraints
- **Interaction Context**: Implementation workflow and validation gates
- **Response Context**: Expected outputs and quality criteria

### Blueprint Components
PRP = PRD + Curated Codebase Intelligence + Agent Runbook

- **Product Requirements**: Clear feature specifications and business context
- **Codebase Intelligence**: Existing patterns, dependencies, and technical constraints
- **Implementation Blueprint**: Phase-based tasks with validation gates
- **Testing Strategy**: Comprehensive test coverage and quality gates
- **Performance Targets**: Measurable success criteria and benchmarks
- **Validation Framework**: Self-correcting mechanisms and error handling

### Planning Integration
- TodoWrite integration for task tracking
- State-based plan lifecycle (draft → ready → pending → completed)
- Git branch strategy for plan execution
- Systematic phase-based implementation

## Integration with EOL Framework

### Context Engineering Implementation
- **Global Rules**: `.claude/CLAUDE.md` provides system-wide context
- **Domain Knowledge**: `.claude/context/` contains specialized patterns
- **Technology Context**: RAG, Redis, Python async patterns, and best practices
- **Project Intelligence**: Existing codebase patterns and conventions
- **Validation Framework**: Quality gates and testing requirements

### Supports Development Workflow
- Feature planning with checkbox format
- Code review and quality assurance
- Virtual environment management
- Documentation and knowledge sharing

## Command Documentation

Each command includes comprehensive help documentation:

- **analyze.md** - Codebase analysis and RAG pattern extraction
- **create.md** - PRP generation with Python/RAG focus
- **execute.md** - Systematic implementation with validation
- **validate.md** - Quality assurance and compliance checking
- **review.md** - Review and update existing PRPs

## Technology-Specific Adaptations

### Python Development Context
- Type hints and async patterns
- Pytest testing frameworks
- Code formatting with Black/isort
- Package management with uv
- Dataclasses and protocols for structure

### RAG Framework Context (Example Domain)
- Document processing and chunking strategies
- Embedding generation and management
- Vector search optimization with Redis
- Semantic caching patterns
- Performance benchmarking and validation

### Context Engineering Principles
- **Codebase Intelligence**: Extract existing patterns and conventions
- **Technical Constraints**: Include dependencies, performance requirements
- **Validation Gates**: Define quality checks and testing requirements
- **Implementation Guidance**: Provide step-by-step execution blueprints
- **Self-Correction**: Enable AI to identify and fix implementation issues

## Benefits

### Development Efficiency
- **Production-Ready Code**: First-pass implementations with comprehensive context
- **Reduced Debugging**: Self-correcting mechanisms catch issues early
- **Faster Delivery**: Systematic approach eliminates trial-and-error development
- **Consistent Quality**: Leveraging proven patterns ensures maintainability

### Code Quality
- **Python Best Practices**: Type hints and async patterns
- **Testing Excellence**: Pytest patterns with high coverage
- **Redis Integration**: Proven vector database patterns

### Team Productivity
- **Knowledge Sharing**: PRPs capture and distribute implementation wisdom
- **Scalable Expertise**: Junior developers can produce senior-level code
- **Onboarding**: New team members learn from comprehensive examples
- **Consistency**: Standardized patterns across all feature development

## File Organization

```
.claude/commands/prp/
├── README.md            # This overview
├── analyze.md          # RAG pattern analysis
├── create.md           # PRP generation
├── execute.md          # Implementation execution
├── validate.md         # Quality validation
└── review.md           # Review and update PRPs

.claude/plans/           # Generated PRPs and plans
├── draft/              # Initial plans
├── ready/              # Validated plans
├── pending/            # Executing plans
└── completed/          # Archived plans
```

## Success Metrics

### Implementation Quality
- **Performance Targets**: 
  - Document indexing >10 docs/sec
  - Vector search <100ms latency
  - Cache hit rate >31%
- **Test Coverage**: >80% with proper async testing
- **Type Safety**: Full mypy compliance

### Development Efficiency
- **Time to Implementation**: Reduced through RAG pattern reuse
- **Pattern Reuse**: Leveraging existing Redis/RAG implementations
- **Integration Success**: Seamless MCP server integration

### Maintainability
- **PRP Currency**: Regular review cycles maintain accuracy
- **Pattern Evolution**: Commands evolve with RAG best practices
- **Documentation Quality**: Living documentation through examples

## Implementation Notes

This PRP command system provides a comprehensive context engineering methodology for any software development:
- **Universal Application**: Works for any technology stack or domain
- **Context Engineering**: Systematic approach to providing AI comprehensive context
- **Production Focus**: Designed to generate production-ready code on first pass
- **Self-Correcting**: Built-in validation and error recovery mechanisms
- **Scalable**: From simple features to complex system implementations

The PRP system transforms software development from iterative exploration into systematic, high-quality implementation with comprehensive context engineering. As the methodology states: "Context isn't a one-time setup. It's an ongoing discipline."